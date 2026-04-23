import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "../../../services/contracts/models";
import { ChatFollowUpChips } from "./ChatFollowUpChips";

interface ChatMessageListProps {
  messages: ChatMessage[];
  onFollowUp?: (question: string) => void;
  followUpsDisabled?: boolean;
}

const CHUNK_STAGGER_MS = 180;

function getChunks(message: ChatMessage): string[] {
  if (message.chunks && message.chunks.length > 0) {
    return message.chunks;
  }
  return [message.content];
}

export function ChatMessageList(props: ChatMessageListProps): JSX.Element {
  const { messages, onFollowUp, followUpsDisabled } = props;

  const lastAssistantIndex = (() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].role === "assistant") return i;
    }
    return -1;
  })();

  return (
    <ul className="chat-message-list">
      {messages.map((message, messageIdx) => {
        const chunks = getChunks(message);
        const isAssistant = message.role === "assistant";
        const showFollowUps =
          isAssistant &&
          messageIdx === lastAssistantIndex &&
          !!message.followUps &&
          message.followUps.length > 0 &&
          !!onFollowUp;

        return (
          <li
            key={message.id}
            className={`chat-message chat-message-${message.role}`}
            aria-label={isAssistant ? "Assistant" : "User"}
          >
            {/* User messages render as plain text — no markdown, no chunks. */}
            {!isAssistant ? (
              <p>{message.content}</p>
            ) : (
              <div className="chat-message-chunks">
                {chunks.map((chunk, chunkIdx) => (
                  <div
                    key={`${message.id}-chunk-${chunkIdx}`}
                    className="chat-message-chunk"
                    style={{ animationDelay: `${chunkIdx * CHUNK_STAGGER_MS}ms` }}
                  >
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{chunk}</ReactMarkdown>
                  </div>
                ))}

                {showFollowUps ? (
                  <ChatFollowUpChips
                    options={message.followUps ?? []}
                    onPick={(q) => onFollowUp?.(q)}
                    disabled={followUpsDisabled}
                  />
                ) : null}
              </div>
            )}
          </li>
        );
      })}
    </ul>
  );
}
