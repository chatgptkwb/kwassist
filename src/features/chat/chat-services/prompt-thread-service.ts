"use server";
import "server-only";

import { userHashedId, userSession } from "@/features/auth/helpers";
//import { FindAllPrompts } from "@/features/chat/chat-services/prompt-service";
import { uniqueId } from "@/features/common/util";
import { SqlQuerySpec } from "@azure/cosmos";
import { CosmosDBContainer } from "../../common/cosmos-prompt";
import {
  CHAT_THREAD_ATTRIBUTE,
  PromptGPTProps,
  PromptModel,
} from "./models";
/*
export const FindAllPromptThreadForCurrentUser = async () => {
  const container = await CosmosDBContainer.getInstance().getContainer();

  const querySpec: SqlQuerySpec = {
    query:
    "SELECT * FROM root r WHERE r.type=@type AND r.userId=@userId AND r.isDeleted=@isDeleted ORDER BY r.createdAt DESC",
    parameters: [
      {
        name: "@type",
        value: CHAT_THREAD_ATTRIBUTE,
      },
      {
        name: "@userId",
        value: await userHashedId(),
      },
      {
        name: "@isDeleted",
        value: false,
      },
    ],
  };

  const { resources } = await container.items
    .query<PromptModel>(querySpec, {
      partitionKey: await userHashedId(),
    })
    .fetchAll();
  return resources;
};

export const FindPromptThreadByID = async (id: string) => {
  const container = await CosmosDBContainer.getInstance().getContainer();

  const querySpec: SqlQuerySpec = {
    query:
      "SELECT * FROM root r WHERE r.type=@type AND r.userId=@userId AND r.id=@id AND r.isDeleted=@isDeleted",
    parameters: [
      {
        name: "@type",
        value: CHAT_THREAD_ATTRIBUTE,
      },
      {
        name: "@userId",
        value: await userHashedId(),
      },
      {
        name: "@id",
        value: id,
      },
      {
        name: "@isDeleted",
        value: false,
      },
    ],
  };

  const { resources } = await container.items
    .query<PromptModel>(querySpec)
    .fetchAll();

  return resources;
};

export const SoftDeletePromptThreadByID = async (chatThreadID: string) => {
  const container = await CosmosDBContainer.getInstance().getContainer();
  const threads = await FindPromptThreadByID(chatThreadID);

  if (threads.length !== 0) {
    const chats = await FindAllPrompts(chatThreadID);

    chats.forEach(async (chat) => {
      const itemToUpdate = {
        ...chat,
      };
      itemToUpdate.isDeleted = true;
      await container.items.upsert(itemToUpdate);
    });

    threads.forEach(async (thread) => {
      const itemToUpdate = {
        ...thread,
      };
      itemToUpdate.isDeleted = true;
      await container.items.upsert(itemToUpdate);
    });
  }
};

export const EnsurePromptThreadIsForCurrentUser = async (
  chatThreadID: string
) => {
  const modelToSave = await FindPromptThreadByID(chatThreadID);
  if (modelToSave.length === 0) {
    throw new Error("Prompt thread not found");
  }

  return modelToSave[0];
};

*/
/*
export const UpsertPrompt = async (prompt: PromptModel) => {
  const container = await CosmosDBContainer.getInstance().getContainer();
  const updatedChatThread = await container.items.upsert<PromptModel>(
    prompt
  );

  if (updatedChatThread === undefined) {
    throw new Error("prompt not found");
  }

  return updatedChatThread;
};

export const updatePromptTitle = async (
  prompt: PromptModel,
  messages: ChatMessageModel[],
  chatType: ChatType,
  conversationStyle: ConversationStyle,
  chatAPIModel: ChatAPIModel,
  chatOverFileName: string,
  userMessage: string
) => {
  if (messages.length === 0) {
    const updatedChatThread = await UpsertPrompt({
      ...prompt,
    });

    return updatedChatThread.resource!;
  }

  return prompt;
};
*/
export const CreatePromptThread = async (dept:string,title:string,content:string) => {
  const modelToSave: PromptModel = {
    id: uniqueId(),
    createdAt: new Date(),
    isDeleted: false,
    userId: await userHashedId(),
    useName: (await userSession())!.name,
    title: title,
    dept: dept,
    threadId: "", // Add the missing property threadId
    content: content // Add the missing property content
  };

  const container = await CosmosDBContainer.getInstance().getContainer();
  const response = await container.items.create<PromptModel>(modelToSave);
  return response.resource;
};



/*
export const initAndGuardPromptSession = async (props: PromptGPTProps) => {
  const { messages, id, chatType, conversationStyle,chatAPIModel, chatOverFileName } = props;

  //last message
  const lastHumanMessage = messages[messages.length - 1];

  const currentPromptThread = await EnsurePromptThreadIsForCurrentUser(id);
  const chats = await FindAllPrompts(id);

  const chatThread = await updatePromptThreadTitle(
    currentPromptThread,
    chats,
    chatType,
    conversationStyle,
    chatAPIModel,
    chatOverFileName,
    lastHumanMessage.content
  );

  return {
    id,
    lastHumanMessage,
    chats,
    chatThread,
  };
};*/
