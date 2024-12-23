import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { FileText } from "lucide-react";
import { ChatFileUI_doc } from "./chat-file-ui-doc";

export const ChatFileSlider_doc = () => {
  return (
    <div>
      <Sheet>
        <SheetTrigger asChild>
          <Button size="icon" variant={"ghost"}>
            <FileText size={16} />
          </Button>
        </SheetTrigger>
        <SheetContent>
          <SheetHeader>
            <SheetTitle>Upload File</SheetTitle>
          </SheetHeader>
          <div className="py-4">
            <ChatFileUI_doc />
          </div>
        </SheetContent>
      </Sheet>
    </div>
  );
};
