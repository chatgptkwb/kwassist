import { Message } from "ai";

export const CHAT_DOCUMENT_ATTRIBUTE = "CHAT_DOCUMENT";
export const CHAT_THREAD_ATTRIBUTE = "CHAT_THREAD";
export const MESSAGE_ATTRIBUTE = "CHAT_MESSAGE";

export interface ChatMessageModel {
  id: string;
  createdAt: Date;
  isDeleted: boolean;
  threadId: string;
  userId: string;
  content: string;
  role: ChatRole;
  context: string;
  type: "CHAT_MESSAGE";
}

export type ConversationStyle = "creative" | "balanced" | "precise";
export type ChatType = "simple" | "web" | "document" | "data" | "doc" | "mssql";
export type ChatAPIModel = "GPT-3" | "GPT-4";
export type ChatDoc = "all" | "latest" | "past" ;

export type ChatRole = "system" | "user" | "assistant" | "function";

export interface ChatThreadModel {
  id: string;
  name: string;
  createdAt: Date;
  userId: string;
  useName: string;
  isDeleted: boolean;
  chatType: ChatType;
  chatDoc: ChatDoc;
  conversationStyle: ConversationStyle;
  chatAPIModel: ChatAPIModel;
  chatOverFileName: string;
  type: "CHAT_THREAD";
}

export interface PromptModel {
  id: string; 
  createdAt: Date;
  userId: string;
  useName: string;
  isDeleted: boolean;
  threadId: string;
  title: string;
  content: string;
  dept: string;
}

export interface PromptGPTBody {
  id: string; // thread id
  chatType: ChatType;
  chatDoc: ChatDoc;
  conversationStyle: ConversationStyle;
  chatAPIModel:ChatAPIModel;
  chatOverFileName: string;
}


export interface PromptGPTProps extends PromptGPTBody {
  messages: Message[];
}

export interface ChatDocumentModel {
  id: string;
  name: string;
  chatThreadId: string;
  userId: string;
  isDeleted: boolean;
  createdAt: Date;
  type: "CHAT_DOCUMENT";
}

export interface ServerActionResponse<T> {
  success: boolean;
  error: string;
  response: T;
}


export interface PromptList {
  id: string;
  title: string;
  content: string;
  dept: string;
/*  createdAt: Date;
  userId: string;
  useName: string;
  isDeleted: boolean;*/
}