import { userHashedId } from "@/features/auth/helpers";
import { OpenAIInstance } from "@/features/common/openai";
import { OpenAIStream, StreamingTextResponse } from "ai";
import { similaritySearchVectorWithScore } from "./azure-cog-search/azure-cog-vector-store";
import { initAndGuardChatSession } from "./chat-thread-service";
import { CosmosDBChatMessageHistory } from "./cosmosdb/cosmosdb";
import { PromptGPTProps } from "./models";

const SYSTEM_PROMPT = `あなたは ${process.env.NEXT_PUBLIC_AI_NAME}です。

回答の際は以下のガイドラインに従ってください：

1. 回答は常に詳細かつ構造化された形式で提供してください：
   - 重要なポイントは箇条書きで説明
   - 比較や対照が必要な場合は表形式を使用
   - 手順や過程は番号付きリストで説明
   - 専門用語は適切に解説

2. 回答の構成：
   - 概要説明（結論から先に述べる）
   - 詳細な説明（複数の観点から）
   - 具体例や参考情報
   - 注意点やリスク（該当する場合）

3. 文体と形式：
   - 丁寧な日本語で説明
   - 簡潔だが必要十分な情報を含める
   - 専門的な内容は可能な限り分かりやすく説明
   - 重要な点は太字や斜体を使用して強調

4. すべての情報源を必ず引用してください：
   - 参照したすべての文書をcitationに含める
   - 情報の出典を明確に示す
   - 不確かな情報は明示する\n`;

const CONTEXT_PROMPT = ({
  context,
  userQuestion,
}: {
  context: string;
  userQuestion: string;
}) => {
  return `
以下の文書から、質問に対する包括的な回答を作成してください。

回答の要件：
- 検索結果の関連情報をすべて含めて回答を作成してください
- 不明な点がある場合は、その旨を明確に伝えてください
- すべての参照文書を必ずcitationに含めてください
- 回答の最後にcitationを必ず含めてください
- citation形式: {% citation items=[{name:"filename",id:"file id"}] /%}
- citationの後に余分なテキストや句読点を追加しないでください

回答の構造：
1. 概要 / 結論
2. 詳細説明
   - 主要ポイント
   - 関連する重要情報
   - 具体例や事例
3. 追加の参考情報（該当する場合）
4. 注意点やリスク（該当する場合）

----------------\n 
コンテキスト情報：\n 
${context}
----------------\n 
質問: ${userQuestion}`;
};

export const ChatAPIDoc = async (props: PromptGPTProps) => {
  const { lastHumanMessage, id, chatThread } = await initAndGuardChatSession(
    props
  );

  const openAI = OpenAIInstance();
  const userId = await userHashedId();

  let chatAPIModel = "";
  if (props.chatAPIModel === "GPT-3") {
    chatAPIModel = "gpt-35-turbo-16k";
  } else {
    chatAPIModel = "gpt-4o";
  }
  chatAPIModel = "gpt-4o-mini";

  const chatHistory = new CosmosDBChatMessageHistory({
    sessionId: chatThread.id,
    userId: userId,
  });

  const chatDoc = props.chatDoc;
  const history = await chatHistory.getMessages();
  const topHistory = history.slice(history.length - 30, history.length);

  // 関連文書の検索数を増やし、より多くの文脈を提供
  const relevantDocuments = await findRelevantDocuments(
    lastHumanMessage.content,
    chatDoc
  );

  // コンテキスト情報の構造化
  const context = relevantDocuments
    .map((result) => {
      const content = result.pageContent.replace(/(\r\n|\n|\r)/gm, "");
      if (!result.source || !result.deptName) {
        console.error('Missing required fields:', { source: result.source, deptName: result.deptName });
        return null;
      }
      const context = `${result.source}\nfile name: ${result.deptName}\nfile id: ${result.id}\n${content}`;
      return context;
    })
    .filter(Boolean) // null値を除外
    .join("\n------\n");

  // citationデータの準備
  const citationData = relevantDocuments
    .filter(doc => doc.source && doc.deptName) // sourceとdeptNameの両方が存在するものだけを使用
    .map(doc => ({
      name: doc.deptName,  // deptNameは必須
      id: doc.id,
      source: doc.source   // sourceも保持
    }));

  if (citationData.length === 0) {
    console.error('No valid documents with both source and deptName found');
  }

  // デバッグ用にcitationDataの内容をログ出力
  console.log('Citation data prepared:', citationData);

  try {
    const response = await openAI.chat.completions.create({
      messages: [
        {
          role: "system",
          content: SYSTEM_PROMPT,
        },
        ...topHistory,
        {
          role: "user",
          content: CONTEXT_PROMPT({
            context,
            userQuestion: lastHumanMessage.content,
          }),
        },
        {
          role: "assistant",
          content: `参照すべき文書情報: ${JSON.stringify(citationData)}`,
        }
      ],
      model: chatAPIModel,
      stream: true,
      temperature: 0.7,
      max_tokens: 4000, // より長い回答を可能に
    });

    // カスタムトランスフォーマーの実装
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();
    let buffer = '';

    const customTransformer = new TransformStream({
      transform(chunk, controller) {
        // デコード処理
        const text = typeof chunk === 'string' ? chunk : decoder.decode(chunk instanceof Uint8Array ? chunk : encoder.encode(chunk.toString()), { stream: true });
        buffer += text;

        // citation形式のチェックと処理
        if (buffer.includes("{% citation")) {
          if (!buffer.includes("/%}")) {
            return; // 完全なcitationブロックを待つ
          }

          // 完全なcitationブロックが見つかった場合
          const parts = buffer.split(/{%\s*citation/);
          if (parts.length > 1) {
            // 最初の部分を送信
            if (parts[0]) {
              controller.enqueue(encoder.encode(parts[0]));
            }

            // citationを含む部分を処理
            const citationPart = parts[1];
            const citationEnd = citationPart.indexOf("/%}") + 3;
            if (citationEnd > 2) {
              // citationDataの内容を検証
              const validCitationData = citationData.map(item => ({
                name: item.name || 'Unknown Document',
                id: item.id
              }));

              // 空のcitationを防ぐための検証
              if (validCitationData.length === 0) {
                console.error('No valid citation data available');
                validCitationData.push({
                  name: 'Document Not Found',
                  id: 'unknown'
                });
              }

              // 既存のcitationを新しいものに置き換え
              const newCitation = `{% citation items=${JSON.stringify(validCitationData)} /%}`;
              controller.enqueue(encoder.encode(newCitation));

              // 残りのテキストをバッファに保存
              buffer = citationPart.slice(citationEnd);
              if (buffer) {
                controller.enqueue(encoder.encode(buffer));
                buffer = '';
              }
            }
          }
        } else {
          // citationを含まない通常のテキスト
          controller.enqueue(encoder.encode(text));
          buffer = '';
        }
      },
      flush(controller) {
        // 残りのバッファをフラッシュ
        if (buffer) {
          controller.enqueue(encoder.encode(buffer));
        }
      }
    });

    const stream = OpenAIStream(response, {
      async onCompletion(completion) {
        await chatHistory.addMessage({
          content: lastHumanMessage.content,
          role: "user",
        });

        await chatHistory.addMessage(
          {
            content: completion,
            role: "assistant",
          },
          context
        );
      },
    }).pipeThrough(customTransformer);

    return new StreamingTextResponse(stream);
  } catch (e: unknown) {
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

/*
const findRelevantDocuments = async (query: string, chatThreadId: string) => {
  // 検索結果の数を増やし、より多くの関連文書を取得
  const relevantDocuments = await similaritySearchVectorWithScore(query, 15, {
    filter: `chatType eq 'doc' `,
  });
  
  // 検索結果をそのまま返す（すべての関連文書を含める）
  return relevantDocuments;
};*/

const findRelevantDocuments = async (query: string, chatDoc: string) => {
  // パラメータ名をchatDocに変更し、型も文字列として明示
  //const chatDoc: string = chatThreadId;

  // 元のfilter条件を復元
  const filter = chatDoc === 'all'
    ? "chatType eq 'doc'"
    : `chatType eq 'doc' and deptName eq '${chatDoc}'`;

  // 検索結果数を10に戻す
  const relevantDocuments = await similaritySearchVectorWithScore(query, 15, {
    filter: filter,
  });
  
  return relevantDocuments;
};