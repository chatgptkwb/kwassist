import Typography from "@/components/typography";
import { Card } from "@/components/ui/card";
import { FC } from "react";
import { useChatContext } from "../chat-context";
import { ChatFileUI } from "../chat-file/chat-file-ui";
import { ChatFileUI_doc } from "../chat-file/chat-file-ui-doc";
import { ChatStyleSelector } from "./chat-style-selector";
import { ChatTypeSelector } from "./chat-type-selector";
import { ChatDeptSelector } from "./chat-dept-seelctor";
import { ChatAPISelector } from "./chat-api-selector";
import { useSession } from "next-auth/react";

interface Prop {}

export const ChatMessageEmptyState: FC<Prop> = (props) => {
  const { fileState } = useChatContext();
  const { data: session } = useSession();

  const { showFileUpload } = fileState;

  return (
    <div className="grid grid-cols-1 w-full items-center container mx-auto max-w-4xl justify-center h-full gap-9">
      <Card className="col-span-3 flex flex-col gap-5 p-5 ">
        <Typography variant="h4" className="text-primary">
        {process.env.NEXT_PUBLIC_AI_NAME}
        </Typography>
          <p className="text-xs text-muted-foreground">
            {process.env.NEXT_PUBLIC_AI_WELCOME_MSG_JA}
            ご利用の際は、以下の項目を選択してください。

          </p>


        <div className="flex flex-col gap-2">
          <p className="text-sm text-muted-foreground">
          会話スタイルを選択してください。
          </p>
          <ChatStyleSelector disable={false} />
        </div>
        <div className="flex flex-col gap-2">
          <p className="text-sm text-muted-foreground">
            AIがお手伝いする方法を選択してください。
          </p>
          <ChatTypeSelector disable={false} />
        </div>
        {(showFileUpload === "doc") && <ChatDeptSelector disable={false} />} 
        {(showFileUpload === "data") && <ChatFileUI />} 
       {/* {((showFileUpload === "doc") && session?.user?.isAdmin) && <ChatFileUI_doc />}  */}
       <div className="flex flex-col gap-2">
        </div>

      </Card>
    </div>
  );
};
