interface ChatFollowUpChipsProps {
  options: string[];
  onPick: (question: string) => void;
  disabled?: boolean;
}

/**
 * Renders a wrap-flex row of clickable chips under the last assistant
 * message. Each chip, when clicked, fires the corresponding follow-up
 * question as if the user had typed it.
 */
export function ChatFollowUpChips(props: ChatFollowUpChipsProps): JSX.Element | null {
  const { options, onPick, disabled } = props;

  if (!options || options.length === 0) {
    return null;
  }

  return (
    <div className="chat-followups" role="list" aria-label="Suggested follow-up questions">
      {options.map((question, idx) => (
        <button
          key={`${idx}-${question.slice(0, 24)}`}
          type="button"
          role="listitem"
          className="chat-followup-chip"
          disabled={disabled}
          onClick={() => onPick(question)}
          aria-label={`Suggested follow-up: ${question}`}
        >
          {question}
        </button>
      ))}
    </div>
  );
}
