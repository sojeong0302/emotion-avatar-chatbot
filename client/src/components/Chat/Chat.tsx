import { useEffect, useMemo, useRef, useState } from "react";
import type { ChatProps } from "./Chat.types";

type Sender = "user" | "bot";

interface Message {
  id: string;
  sender: Sender;
  text: string;
  createdAt: number;
}

function makeId() {
  // 간단 ID (데모용). 실제론 uuid 써도 됨.
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export default function Chat({ title = "채팅" }: ChatProps) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>(() => [
    {
      id: makeId(),
      sender: "bot",
      text: "안녕! 뭐든 입력해봐 🙂",
      createdAt: Date.now(),
    },
  ]);

  const bottomRef = useRef<HTMLDivElement | null>(null);

  const canSend = useMemo(() => input.trim().length > 0, [input]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  const sendMessage = () => {
    const text = input.trim();
    if (!text) return;

    const userMsg: Message = {
      id: makeId(),
      sender: "user",
      text,
      createdAt: Date.now(),
    };

    const botMsg: Message = {
      id: makeId(),
      sender: "bot",
      text: "응, 맞아",
      createdAt: Date.now() + 1,
    };

    setMessages((prev) => [...prev, userMsg, botMsg]);
    setInput("");
  };

  const onKeyDown: React.KeyboardEventHandler<HTMLInputElement> = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex h-[700px] w-full flex-col overflow-hidden rounded-2xl bg-white shadow">
      {/* Header */}
      <div className="flex items-center justify-between border-b px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-full bg-gray-200" />
          <div className="leading-tight">
            <div className="text-sm font-semibold text-gray-900">{title}</div>
            <div className="text-xs text-gray-500">온라인</div>
          </div>
        </div>
        <div className="text-xs text-gray-400">Demo</div>
      </div>

      {/* Messages */}
      <div className="flex-1 space-y-3 overflow-y-auto bg-gray-50 px-3 py-4">
        {messages.map((m) => (
          <MessageBubble key={m.id} sender={m.sender} text={m.text} />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t bg-white p-3">
        <div className="flex items-end gap-2">
          <div className="flex-1 rounded-2xl border bg-gray-50 px-3 py-2 focus-within:ring-2 focus-within:ring-blue-200">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="메시지 입력"
              className="w-full bg-transparent text-sm text-gray-900 outline-none placeholder:text-gray-400"
            />
          </div>

          <button
            type="button"
            onClick={sendMessage}
            disabled={!canSend}
            className="rounded-2xl bg-blue-500 px-4 py-2 text-sm font-semibold text-white shadow disabled:cursor-not-allowed disabled:opacity-40"
          >
            전송
          </button>
        </div>

        <div className="mt-2 text-[11px] text-gray-400">Enter로 전송</div>
      </div>
    </div>
  );
}

function MessageBubble({ sender, text }: { sender: Sender; text: string }) {
  const isUser = sender === "user";

  return (
    <div className={isUser ? "flex justify-end" : "flex justify-start"}>
      <div className={isUser ? "max-w-[78%]" : "flex max-w-[78%] gap-2"}>
        {!isUser && (
          <div className="mt-1 h-8 w-8 flex-shrink-0 rounded-full bg-gray-200" />
        )}

        <div
          className={[
            "rounded-2xl px-3 py-2 text-sm leading-relaxed shadow-sm",
            isUser
              ? "rounded-br-md bg-blue-500 text-white"
              : "rounded-bl-md bg-white text-gray-900",
          ].join(" ")}
        >
          {text}
        </div>
      </div>
    </div>
  );
}
