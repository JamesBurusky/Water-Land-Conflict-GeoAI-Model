import type { ReactNode } from "react";
import { useWidgetStore, type WidgetId } from "../store/useWidgetStore";

export function Widget({ id, title, children }: { id: WidgetId; title: string; children: ReactNode }) {
  const visible = useWidgetStore((s) => s.visible[id]);
  const toggle = useWidgetStore((s) => s.toggle);

  if (!visible) return null;

  return (
    <div className="widget">
      <div className="widget-header">
        <span>{title}</span>
        <button className="widget-close" onClick={() => toggle(id)} aria-label={`Close ${title}`}>
          ×
        </button>
      </div>
      <div className="widget-body">{children}</div>
    </div>
  );
}
