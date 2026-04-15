import { useEffect, useRef, type ReactNode } from "react";

export type LayoutMode = "upload" | "split" | "merged" | "results";

interface PanelTransitionWrapperProps {
  mode: LayoutMode;
  className?: string;
  left: ReactNode;
  right: ReactNode;
}

export function PanelTransitionWrapper(
  props: PanelTransitionWrapperProps
): JSX.Element {
  const { mode, className, left, right } = props;
  const prevMode = useRef(mode);

  /* scroll to top when layout mode changes (e.g. upload → split → results) */
  useEffect(() => {
    if (prevMode.current !== mode) {
      prevMode.current = mode;
      window.scrollTo({ top: 0, behavior: "instant" });
    }
  }, [mode]);

  return (
    <section
      className={`menu-grid panel-transition-wrapper layout-${mode} ${className ?? ""}`.trim()}
    >
      <div className="left-column transition-panel-left">{left}</div>
      <div className="right-column transition-panel-right">{right}</div>
    </section>
  );
}
