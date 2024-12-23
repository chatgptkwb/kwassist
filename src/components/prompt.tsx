import * as React from "react";

import { cn } from "@/lib/utils";
import Link from "next/link";

const Prompt = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("w-80 flex flex-col", className)} {...props} />
));

Prompt.displayName = "Prompt";

const PromptHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex pb-2 justify-between items-center", className)}
    {...props}
  />
));
PromptHeader.displayName = "PromptHeader";

const PromptContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "flex flex-col flex-1 overflow-y-auto py-2 gap-0.5",
      className
    )}
    {...props}
  />
));
PromptContent.displayName = "PromptContent";

interface PromptItemProps extends React.HTMLAttributes<HTMLLinkElement> {
  href: string;
  isSelected?: boolean;
}

const PromptItem: React.FC<PromptItemProps> = (props) => {
  return (
    <Link
      className={cn(
        props.className,
        "items-center text-sm font-medium flex gap-2 p-2 py-1 rounded-md hover:bg-primary/25",
        props.isSelected && "bg-primary/25"
      )}
      href={props.href}
    >
      {props.children}
    </Link>
  );
};

const PromptFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("flex flex-col", className)} {...props} />
));
PromptFooter.displayName = "PromptFooter";

export { Prompt, PromptContent, PromptFooter, PromptHeader, PromptItem };
