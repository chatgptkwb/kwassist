import { userHashedId } from "@/features/auth/helpers";
import { OpenAIInstance } from "@/features/common/openai";
import { OpenAIStream, StreamingTextResponse } from "ai";
import { initAndGuardChatSession } from "./chat-thread-service";
import { CosmosDBChatMessageHistory } from "./cosmosdb/cosmosdb";
import { BingSearchResult } from "./Azure-bing-search/bing";
import { PromptGPTProps } from "./models";
import puppeteer from 'puppeteer';

export const ChatAPIWeb = async (props: PromptGPTProps) => {
  // Destructure and initialize variables
  const { lastHumanMessage, chatThread } = await initAndGuardChatSession(props);
  const openAI = OpenAIInstance();
  const userId = await userHashedId();

  // Select appropriate model
  let chatAPIModel = props.chatAPIModel === "GPT-3" ? "gpt-35-turbo-16k" : "gpt-4o-mini";

  // Initialize Bing Search
  const bing = new BingSearchResult();
  const searchResult = await bing.SearchWeb(lastHumanMessage.content);

  // Enhanced web page content extraction
  const webPageContents = await Promise.all(
    searchResult.webPages.value.slice(0, 5).map(async (page: any) => {
      try {
        // Use Puppeteer to scrape page content
        const browser = await puppeteer.launch({ headless: true });
        const pageInstance = await browser.newPage();
        await pageInstance.goto(page.url, { waitUntil: 'networkidle0' });
        
        // Extract main content, avoiding scripts, styles, etc.
        const pageText = await pageInstance.evaluate(() => {
          // Remove script, style, and other non-content elements
          const scripts = document.getElementsByTagName('script');
          const styles = document.getElementsByTagName('style');
          Array.from(scripts).forEach(script => script.remove());
          Array.from(styles).forEach(style => style.remove());
          
          // Try to extract main content
          const mainContent = 
            document.querySelector('main')?.innerText || 
            document.querySelector('article')?.innerText || 
            document.body.innerText;
          
          return mainContent || '';
        });

        await browser.close();

        return {
          url: page.url,
          title: page.name,
          snippet: page.snippet,
          content: pageText.substring(0, 2000) // Limit content length
        };
      } catch (error) {
        console.error(`Error scraping ${page.url}:`, error);
        return {
          url: page.url,
          title: page.name,
          snippet: page.snippet,
          content: page.snippet
        };
      }
    })
  );

  // Construct comprehensive prompt
  const Prompt = `
問い合わせ: ${lastHumanMessage.content}

Web検索結果の概要:
${webPageContents.map(page => 
  `タイトル: ${page.title}
URL: ${page.url}
スニペット: ${page.snippet}

詳細コンテンツ抜粋:
${page.content.substring(0, 500)}...
`).join('\n\n')}

上記の検索結果を踏まえて、元の質問に対して包括的かつ情報豊富な回答を2000文字程度で生成してください。
`;

  // Initialize chat history
  const chatHistory = new CosmosDBChatMessageHistory({
    sessionId: chatThread.id,
    userId: userId,
  });

  // Add user message to chat history
  await chatHistory.addMessage({
    content: lastHumanMessage.content,
    role: "user",
  });

  // Get recent chat history
  const history = await chatHistory.getMessages();
  const topHistory = history.slice(history.length - 30, history.length);

  try {
    // Create OpenAI chat completion
    const response = await openAI.chat.completions.create({
      messages: [
        {
          role: "system",
          content: `ユーザーからの質問に対して日本語で丁寧に回答します。
          - 質問には正直かつ正確に答えます。
          - Web検索結果を参考にしつつ、信頼性の高い情報を提供します。
          - 情報の出典を適切に示します。`,
        },
        {
          role: "user",
          content: Prompt,
        }
      ],
      model: chatAPIModel,
      stream: true,
      max_tokens: 2000, // Limit response length
      temperature: 0.7, // Balanced creativity and factuality
    });

    // Stream the response
    const stream = OpenAIStream(response, {
      async onCompletion(completion) {
        await chatHistory.addMessage({
          content: completion,
          role: "assistant",
        });
      },
    });

    return new StreamingTextResponse(stream);
    
  } catch (e: unknown) {
    // Error handling
    if (e instanceof Error) {
      return new Response(e.message, {
        status: 500,
        statusText: e.toString(),
      });
    } else {
      return new Response("An unknown error occurred.", {
        status: 500,
        statusText: "Unknown Error",
      });
    }
  }
};