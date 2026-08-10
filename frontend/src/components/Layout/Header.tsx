import { NavLink, useLocation } from "react-router-dom";
import { Camera, Clock, Sparkles } from "lucide-react";

export default function Header() {
  const location = useLocation();

  const links = [
    { to: "/", label: "工作台", icon: Camera },
    { to: "/history", label: "历史记录", icon: Clock },
  ];

  return (
    <header className="sticky top-0 z-50 border-b border-dark-700/50 bg-dark-950/80 backdrop-blur-xl">
      <div className="mx-auto flex h-14 max-w-[1920px] items-center justify-between px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-amber-accent to-amber-dark">
            <Sparkles className="h-4 w-4 text-dark-950" />
          </div>
          <h1 className="font-display text-lg font-semibold tracking-wide text-ivory">
            时光修复工坊
          </h1>
          <span className="hidden text-xs text-ivory-dim sm:inline">
            PhotoRevive
          </span>
        </div>

        <nav className="flex items-center gap-1">
          {links.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={() => {
                const isActive = location.pathname === to;
                return `flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? "bg-amber-accent/10 text-amber-accent"
                    : "text-ivory-muted hover:bg-dark-700/50 hover:text-ivory"
                }`;
              }}
            >
              <Icon className="h-4 w-4" />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}
