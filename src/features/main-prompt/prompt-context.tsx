import React, { createContext, useContext, useState } from "react";

interface PromptContextProps {
  isPromptOpen: boolean;
  togglePrompt: () => void;
}

export const PromptContext = createContext<PromptContextProps>({
  isPromptOpen: true,
  togglePrompt: () => {},
});

export const PromptProvider = ({ children }: { children: React.ReactNode }) => {
  const [isPromptOpen, setIsPromptOpen] = useState(true);

  const togglePrompt = () => {
    setIsPromptOpen(!isPromptOpen);
  };

  return (
    <PromptContext.Provider value={{ isPromptOpen, togglePrompt }}>
      {children}
    </PromptContext.Provider>
  );
};

export const usePromptContext = () => useContext(PromptContext);
