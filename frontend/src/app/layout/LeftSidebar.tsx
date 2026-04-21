import { useAuth } from "../AuthContext";
import { UserAvatarButton } from "./UserAvatarButton";

export function LeftSidebar(): JSX.Element {
  const { signOut } = useAuth();
  const pathname = typeof window !== "undefined" ? window.location.pathname : "/";
  const inLabeling = pathname.startsWith("/labeling");

  const handleMenuClick = (): void => {
    window.location.href = inLabeling ? "/" : "/labeling";
  };

  const handleSignOut = (): void => {
    if (window.confirm("Sign out?")) {
      void signOut();
    }
  };

  return (
    <aside className="left-sidebar" aria-label="Sidebar">
      <nav className="sidebar-nav" aria-label="Main navigation">
        <a
          href="/"
          className={`sidebar-nav-link ${!inLabeling ? "sidebar-nav-link-active" : ""}`}
          aria-label="Open main app"
        >
          <span>App</span>
        </a>
        <a
          href="/labeling"
          className={`sidebar-nav-link ${inLabeling ? "sidebar-nav-link-active" : ""}`}
          aria-label="Open labeling workspace"
        >
          <span>Labels</span>
        </a>
      </nav>

      <button
        type="button"
        className="sidebar-icon-button"
        aria-label={inLabeling ? "Go to App" : "Go to Labels"}
        title={inLabeling ? "Go to App" : "Go to Labels"}
        onClick={handleMenuClick}
      >
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
        >
          <rect
            x="3"
            y="4"
            width="18"
            height="4"
            rx="1"
            stroke="currentColor"
            strokeWidth="1.8"
          />
          <path
            d="M6 8V18H18V8"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M9 12H15"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
          />
        </svg>
      </button>

      <UserAvatarButton
        className="sidebar-avatar-button"
        onClick={handleSignOut}
        title="Sign out"
      />
    </aside>
  );
}
