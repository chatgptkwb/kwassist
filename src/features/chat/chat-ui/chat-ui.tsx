"use client";

import { FC } from "react";
import { useState } from "react";
import { useChatContext } from "./chat-context";
import { ChatMessageEmptyState } from "./chat-empty-state/chat-message-empty-state";
import ChatPromptEmptyState from "./chat-prompt/chat-prompt-empty-state";
import ChatInput from "./chat-input/chat-input";
import { ChatMessageContainer } from "./chat-message-container";
interface Prop {}
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import {
  MessageCircle,
  Lightbulb,
} from "lucide-react";


import Box from '@mui/material/Box';
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}
function CustomTabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}
function a11yProps(index: number) {
  return {
    id: `simple-tab-${index}`,
    'aria-controls': `simple-tabpanel-${index}`,
  };
}

export const ChatUI: FC<Prop> = () => {
  const { messages } = useChatContext();
  const [value, setValue] = useState(0);
  const handleChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };

  return (
    <div className="h-full relative overflow-auto flex-1 bg-card rounded-md shadow-md">
    <Tabs value={value} onChange={handleChange} aria-label="設定画面">
      <Tab icon={<MessageCircle />} label="チャット画面" {...a11yProps(0)} />
      <Tab icon={<Lightbulb />} label="プロンプト集" {...a11yProps(1)} />
    </Tabs>

  <CustomTabPanel value={value} index={0}>
      {messages.length !== 0 ? (
        <ChatMessageContainer />
      ) : (
        <ChatMessageEmptyState />
      )}
    <div className="flex flex-col h-full overflow-auto">
      {/* Other content here */}
      <div className="mt-auto">
      <br/><br/><br/>
        <ChatInput />
      </div>
    </div>   
  </CustomTabPanel>


  <CustomTabPanel value={value} index={1}>
    <ChatPromptEmptyState />
  </CustomTabPanel>
</div>
  );
};
