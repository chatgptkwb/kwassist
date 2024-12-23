import Typography from "@/components/typography";
import { Card } from "@/components/ui/card";
import { FC } from "react";
import { NewChat } from "../../chat-menu/new-chat";

interface Prop {}

export const StartNewChat: FC<Prop> = (props) => {
  return (
    <div className="grid grid-cols-5 w-full items-center container mx-auto max-w-3xl justify-center h-full gap-9">
      <div className="col-span-2 gap-5 flex flex-col flex-1">
        <img src="/ai-icon.png" className="w-36" />
      </div>
      <Card className="col-span-3 flex flex-col gap-5 p-5 ">
        <Typography variant="h4" className="text-primary">
        {process.env.NEXT_PUBLIC_AI_NAME}
        </Typography>
        <div className="flex flex-col gap-2">
          <p className="text-sm text-muted-foreground">
          {process.env.NEXT_PUBLIC_AI_WELCOME_MSG_JA}
          </p>
          <p>下記の➕を押して会話を開始できます</p>
        </div>
        <div className="-mx-1 -mb-1 p-1 flex flex-col border-t bg-muted">
          <NewChat />
        </div>
      </Card>
    </div>
  );
};
