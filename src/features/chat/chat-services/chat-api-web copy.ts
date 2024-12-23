import { userHashedId } from "@/features/auth/helpers";
import { OpenAIInstance } from "@/features/common/openai";
import { OpenAIStream, StreamingTextResponse } from "ai";
import { initAndGuardChatSession } from "./chat-thread-service";
import { CosmosDBChatMessageHistory } from "./cosmosdb/cosmosdb";
import { BingSearchResult } from "./Azure-bing-search/bing";
import { PromptGPTProps } from "./models";
import puppeteer from 'puppeteer'
import { Lexend_Tera } from "next/font/google";

export const ChatAPIWeb = async (props: PromptGPTProps) => {
  var snippet = "";
  var Prompt = "";
  var BingResult = "";
  var WebinnerText = "";
  const { lastHumanMessage, chatThread } = await initAndGuardChatSession(props);

  const openAI = OpenAIInstance();

  const userId = await userHashedId();

  let chatAPIModel = "";
  if (props.chatAPIModel === "GPT-3") {
    chatAPIModel = "gpt-35-turbo-16k";
  }else{
    chatAPIModel = "gpt-4o";
  }

  chatAPIModel = "gpt-4o-mini"
//  console.log("Model_web: ", process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME);
//  console.log("PromptGPTProps_web: ", props.chatAPIModel);

  const bing = new BingSearchResult();
  const searchResult = await bing.SearchWeb(lastHumanMessage.content);

  snippet = searchResult.webPages.value[0].snippet;
  snippet += searchResult.webPages.value[1].snippet;
  snippet += searchResult.webPages.value[2].snippet;
  snippet += searchResult.webPages.value[3].snippet;
  snippet += searchResult.webPages.value[4].snippet; 
  snippet += searchResult.webPages.value[5].snippet; 
  snippet += searchResult.webPages.value[6].snippet; 
  snippet += searchResult.webPages.value[7].snippet; 
  snippet += searchResult.webPages.value[8].snippet; 
  snippet += searchResult.webPages.value[9].snippet; 


/*   // ブラウザを起動
  const browser = await puppeteer.launch({ headless: true })
  // 新しくページを開く
  const page = await browser.newPage()
  // 対象のページへ遷移
  await page.goto(searchResult.webPages.value[0].url)
  let pageText = await page.evaluate(() => document.body.innerText);
  WebinnerText = '参照URL:'+ searchResult.webPages.value[0].url + '検索結果:'+ pageText.substring(0, 1000);
  // 対象のページへ遷移
  //await page.goto(searchResult.webPages.value[1].url)
  //pageText += await page.evaluate(() => document.body.innerText);
  //WebinnerText = '参照URL:'+ searchResult.webPages.value[0].url + '検索結果:'+ pageText.substring(0, 1000);
　// ブラウザを閉じる
  await browser.close()
 */
  BingResult = + searchResult.webPages.value[0].name + "\n" + searchResult.webPages.value[0].snippet + "\n";
  BingResult += + searchResult.webPages.value[1].name + "\n" + searchResult.webPages.value[1].snippet + "\n";
  BingResult += + searchResult.webPages.value[2].name + "\n" + searchResult.webPages.value[2].snippet + "\n";
  BingResult += + searchResult.webPages.value[3].name + "\n" + searchResult.webPages.value[3].snippet + "\n";
  BingResult += + searchResult.webPages.value[4].name + "\n" + searchResult.webPages.value[4].snippet + "\n";

  //console.log(snippet) ;
  Prompt = "次の{問い合わせ}について、{Web検索結果}を元に2000文字程度で回答を生成してください。" ;
  Prompt += "【問い合わせ】 "  + lastHumanMessage.content ;
  //Prompt += "【Web検索結果】" + snippet; 
  Prompt += "【Web検索結果】" + snippet; 
  //Prompt += "参照URLを回答最後に表示してください"; 

  const chatHistory = new CosmosDBChatMessageHistory({
    sessionId: chatThread.id,
    userId: userId,
  });

  await chatHistory.addMessage({
    content: lastHumanMessage.content,
    role: "user",
  });

  const history = await chatHistory.getMessages();
  const topHistory = history.slice(history.length - 30, history.length);
  //var topHistory = "[ { role: 'user', content: '" + Prompt + "' } ]";
  //console.log(topHistory);

  try {
    const response = await openAI.chat.completions.create({
      messages: [
        {
          role: "system",
          content: `ユーザーからの質問に対して日本語で丁寧に回答します。
          - 質問には正直かつ正確に答えます。`,
        },
        {
          role: "user",
          content: Prompt,
        }
      ],
      //model: process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME,
      model: chatAPIModel,
      stream: true,
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
