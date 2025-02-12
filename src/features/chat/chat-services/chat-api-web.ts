import { userHashedId } from "@/features/auth/helpers";
import { OpenAIInstance } from "@/features/common/openai";
import { OpenAIStream, StreamingTextResponse } from "ai";
import { initAndGuardChatSession } from "./chat-thread-service";
import { CosmosDBChatMessageHistory } from "./cosmosdb/cosmosdb";
import { BingSearchResult } from "./Azure-bing-search/bing";
import { PromptGPTProps } from "./models";
import puppeteer from "puppeteer";

// 日付関連の文字列を検出する正規表現
const DATE_PATTERNS = [
  /今日|本日|現在|最新/,
  /(\d{4}年)?(\d{1,2})月(\d{1,2})日/,
  /昨日|一昨日/,
  /今週|今月|今年/,
  /最近|直近/
];

// 日本時間での現在時刻を取得する関数
function getJapanDateTime(): Date {
  return new Date(new Date().toLocaleString("en-US", { timeZone: "Asia/Tokyo" }));
}

// 検索クエリを最適化する関数
function optimizeSearchQuery(message: string, jpDateTime: Date): string {
  let query = message;
  
  // 日付パターンを検出
  const hasDatePattern = DATE_PATTERNS.some(pattern => pattern.test(message));
  
  if (hasDatePattern) {
    // 日付が含まれる場合、検索クエリに年月を追加
    const year = jpDateTime.getFullYear();
    const month = jpDateTime.getMonth() + 1;
    query = `${query} ${year}年${month}月`;
  }
  
  // 「最新」「現在」などのキーワードがある場合、より具体的な期間指定を追加
  if (/最新|現在|今|本日/.test(message)) {
    // 日本時間での日付を YYYY-MM-DD 形式で取得
    const jpDate = jpDateTime.toISOString().split('T')[0];
    query += ` after:${jpDate}`;
  }
  
  return query;
}

export const ChatAPIWeb = async (props: PromptGPTProps) => {
  try {
    const { lastHumanMessage, chatThread } = await initAndGuardChatSession(props);
    const openAI = OpenAIInstance();
    const userId = await userHashedId();

    let chatAPIModel =
      props.chatAPIModel === "GPT-3" ? "gpt-35-turbo-16k" : "gpt-4o-mini";

    const chatHistory = new CosmosDBChatMessageHistory({
      sessionId: chatThread.id,
      userId: userId,
    });

    const history = await chatHistory.getMessages();
    const topHistory = history.slice(history.length - 30, history.length);

    // 日本時間の取得
    const jpDateTime = getJapanDateTime();
    const currentTimeStr = jpDateTime.toLocaleString("ja-JP", { timeZone: "Asia/Tokyo" });

    // 検索クエリの最適化（日本時間ベース）
    const optimizedQuery = optimizeSearchQuery(lastHumanMessage.content, jpDateTime);
    
    // Bing検索の実行（freshness パラメータを日本時間に基づいて設定）
    const bing = new BingSearchResult();
    const searchResult = await bing.SearchWeb(optimizedQuery, {
      freshness: /最新|現在|今|本日/.test(lastHumanMessage.content) ? "Day" : "Week",
      count: 5,
    });

    if (!searchResult?.webPages?.value) {
      console.warn('No web search results found');
      return await handleChatCompletion(
        openAI,
        chatHistory,
        lastHumanMessage,
        topHistory,
        chatAPIModel,
        [],
        currentTimeStr
      );
    }

    // Web検索結果の処理
    const webPageContents = await Promise.all(
      searchResult.webPages.value.map(async (page: any) => {
        try {
          const browser = await puppeteer.launch({ 
            headless: true,
            args: ['--no-sandbox', '--disable-setuid-sandbox']
          });
          const pageInstance = await browser.newPage();
          
          try {
            await pageInstance.goto(page.url, {
              waitUntil: "networkidle0",
              timeout: 30000,
            });

            const pageText = await pageInstance.evaluate(() => {
              const removeElements = (selector: string) => {
                document.querySelectorAll(selector).forEach((el) => el.remove());
              };

              // 不要な要素を除去
              removeElements("script");
              removeElements("style");
              removeElements("nav");
              removeElements("header");
              removeElements("footer");
              removeElements("aside");
              removeElements(".advertisement");
              removeElements(".social-share");

              // メインコンテンツの抽出を試みる
              const contentSelectors = [
                "main",
                "article",
                '[role="main"]',
                "#main-content",
                ".main-content",
                ".content",
                ".post-content",
                "body",
              ];

              for (const selector of contentSelectors) {
                const element = document.querySelector(selector);
                if (element?.textContent) {
                  return element.textContent.trim();
                }
              }

              return document.body.textContent?.trim() || "";
            });

            // 公開日時の取得と日本時間への変換
            const publishDate = await pageInstance.evaluate(() => {
              const dateSelectors = [
                'meta[property="article:published_time"]',
                'meta[name="publication_date"]',
                'time[datetime]',
                '.published-date',
                '.post-date',
                '.article-date'
              ];

              for (const selector of dateSelectors) {
                const element = document.querySelector(selector);
                if (element) {
                  const dateStr = element.getAttribute('content') || 
                                element.getAttribute('datetime') || 
                                element.textContent;
                  if (dateStr) {
                    try {
                      // 日付文字列をパースして日本時間に変換
                      const date = new Date(dateStr);
                      return date.toLocaleString("ja-JP", { timeZone: "Asia/Tokyo" });
                    } catch (e) {
                      return dateStr;
                    }
                  }
                }
              }
              return null;
            });

            await browser.close();

            const cleanUrl = new URL(page.url).toString();
            return {
              url: cleanUrl,
              title: page.name || "Untitled",
              snippet: page.snippet || "",
              content: pageText.substring(0, 2000),
              publishDate: publishDate || 
                         (page.datePublished ? new Date(page.datePublished)
                           .toLocaleString("ja-JP", { timeZone: "Asia/Tokyo" }) : null),
            };
          } catch (error) {
            await browser.close();
            throw error;
          }
        } catch (error) {
          console.error(`Error scraping ${page.url}:`, error);
          return {
            url: page.url,
            title: page.name || "Untitled",
            snippet: page.snippet || "",
            content: page.snippet || "",
            publishDate: page.datePublished ? 
                        new Date(page.datePublished)
                          .toLocaleString("ja-JP", { timeZone: "Asia/Tokyo" }) : null,
          };
        }
      })
    );

    return await handleChatCompletion(
      openAI,
      chatHistory,
      lastHumanMessage,
      topHistory,
      chatAPIModel,
      webPageContents,
      currentTimeStr
    );

  } catch (e: unknown) {
    console.error('ChatAPIWeb error:', e);
    if (e instanceof Error) {
      return new Response(e.message, {
        status: 500,
        statusText: e.toString(),
      });
    }
    return new Response("An unknown error occurred.", {
      status: 500,
      statusText: "Unknown Error",
    });
  }
};

async function handleChatCompletion(
  openAI: any,
  chatHistory: CosmosDBChatMessageHistory,
  lastHumanMessage: any,
  topHistory: any[],
  chatAPIModel: string,
  webPageContents: any[],
  currentTimeStr: string
) {
  await chatHistory.addMessage({
    content: lastHumanMessage.content,
    role: "user",
  });

  const Prompt = `
現在の日本時間: ${currentTimeStr}

以前の会話の文脈:
${topHistory.map((msg) => `${msg.role}: ${msg.content}`).join("\n")}

最新の問い合わせ: ${lastHumanMessage.content}

${webPageContents.length > 0 ? `Web検索結果の概要:
${webPageContents
  .map(
    (page) => `
タイトル: ${page.title}
URL: [${page.url}](${page.url})
${page.publishDate ? `公開日時: ${page.publishDate}` : ''}
スニペット: ${page.snippet}

詳細コンテンツ抜粋:
${page.content.substring(0, 500)}...
`
  )
  .join("\n\n")}` : "Web検索結果はありませんでした。"}

指示:
1. 上記の会話の文脈${webPageContents.length > 0 ? 'と検索結果' : ''}を踏まえて、最新の質問に対して包括的かつ情報豊富な回答を生成してください。
2. 情報の鮮度が重要な質問の場合、各情報源の公開日時を考慮して、最新の情報を優先してください。
3. 情報の時点を明確にするため、可能な限り日付情報を含めてください。
4. 「本日」「現在」などの表現を使用する場合は、具体的な日本時間（${currentTimeStr}）に基づいて回答してください。

${webPageContents.length > 0 ? `回答の最後には、以下の形式でMarkdown形式の参考文献リストを必ず含めてください:

### 参考文献
- [タイトル1](URL1) (公開日時: YYYY-MM-DD HH:MM JST)
- [タイトル2](URL2) (公開日時: YYYY-MM-DD HH:MM JST)` : ''}
`;

  const response = await openAI.chat.completions.create({
    messages: [
      {
        role: "system",
        content: `あなたは ${process.env.NEXT_PUBLIC_AI_NAME} です。ユーザーからの質問に対して日本語で丁寧に回答します。以下の指示に従ってください：

1. 質問には会話の文脈を考慮しながら、正直かつ正確に答えてください。

2. Web検索結果がある場合：
   - 情報の鮮度を重視し、最新の情報を優先して提供してください。
   - 情報源の公開日時を確認し、古い情報は参考程度に扱ってください。
   - 時事的な内容の場合、必ず情報の日付を明記してください。

3. 「今日」「最近」などの相対的な時間表現がある場合：
   - システムから提供された現在の日本時間を基準として具体的な日時に置き換えて情報を提供してください。
   - 情報の時点を明確にしてください。

4. Web検索結果がある場合、回答の最後には必ず「### 参考文献」という見出しを付け、その後に参照元を以下のMarkdown形式で列挙してください：
   - [タイトルテキスト](URL) (公開日時: YYYY-MM-DD HH:MM JST)

5. 以下の点に注意してください：
   - 以前の会話内容と矛盾する情報を提供しない
   - HTMLタグは使用せず、必ずMarkdown記法を使用する
   - 情報の不確かさや制限事項がある場合は、その旨を明記する
   - 日時の表現は必ず日本時間（JST）で行う`,
      },
      ...topHistory,
      {
        role: "user",
        content: Prompt,
      },
    ],
    model: chatAPIModel,
    stream: true,
    max_tokens: 2000,
    temperature: 0.7,
  });

  const stream = OpenAIStream(response, {
    async onCompletion(completion) {
      await chatHistory.addMessage({
        content: completion,
        role: "assistant",
      });
    },
  });

  return new StreamingTextResponse(stream);
}