import { FC } from "react";
import { useChatContext } from "./chat-context";
import { ChatStyleSelector } from "./chat-empty-state/chat-style-selector";
import { ChatTypeSelector } from "./chat-empty-state/chat-type-selector";
import { ChatAPISelector } from "./chat-empty-state/chat-api-selector";

interface Prop {}

export const ChatHeader: FC<Prop> = (props) => {
  const { chatBody } = useChatContext();
  return (
    <div className="flex flex-col gap-3">
      <div className="flex gap-3">
        <ChatTypeSelector disable={true} />
        <ChatStyleSelector disable={true} />
        <ChatAPISelector disable={true} />
      </div>
      <div className="flex gap-2 h-2">
        <p className="text-xs">{chatBody.chatOverFileName}</p>
      </div>
    </div>
  );
};
