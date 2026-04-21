import { useAuth } from "../AuthContext";
import { useNavigation } from "../NavigationContext";
import { UserAvatarButton } from "./UserAvatarButton";

export function LeftSidebar(): JSX.Element {
  const { signOut } = useAuth();
  const { currentRoute, navigate } = useNavigation();
  const inLabeling = currentRoute === "labeling";

  const handleMenuClick = (): void => {
    navigate("labeling");
  };

  const handleSignOut = (): void => {
    if (window.confirm("Sign out?")) {
      void signOut();
    }
  };

  return (
    <aside className="left-sidebar" aria-label="Sidebar">
      <nav className="sidebar-nav" aria-label="Main navigation">
        <button
          type="button"
          className={`sidebar-nav-link ${!inLabeling ? "sidebar-nav-link-active" : ""}`}
          aria-label="Open main app"
          onClick={() => navigate("app")}
        >
          <span>App</span>
        </button>
        <button
          type="button"
          className={`sidebar-nav-link ${inLabeling ? "sidebar-nav-link-active" : ""}`}
          aria-label="Open labeling workspace"
          onClick={() => navigate("labeling")}
        >
          <span>Labels</span>
        </button>
      </nav>

      <button
        type="button"
        className="sidebar-icon-button"
        aria-label="Go to Labels"
        title="Go to Labels"
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
