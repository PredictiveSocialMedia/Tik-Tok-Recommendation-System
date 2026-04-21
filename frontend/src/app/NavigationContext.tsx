import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";

export type AppRoute = "app" | "labeling";

interface NavigationContextValue {
  currentRoute: AppRoute;
  navigate: (route: AppRoute) => void;
}

const NavigationContext = createContext<NavigationContextValue>({
  currentRoute: "app",
  navigate: () => {},
});

function resolveInitialRoute(): AppRoute {
  if (typeof window === "undefined") return "app";
  return window.location.pathname.startsWith("/labeling") ||
    window.location.hash === "#labeling"
    ? "labeling"
    : "app";
}

export function NavigationProvider({ children }: { children: ReactNode }): JSX.Element {
  const [currentRoute, setCurrentRoute] = useState<AppRoute>(resolveInitialRoute);

  const navigate = useCallback((route: AppRoute): void => {
    setCurrentRoute(route);
    const path = route === "labeling" ? "/labeling" : "/";
    window.history.pushState({}, "", path);
  }, []);

  useEffect(() => {
    const onPopState = (): void => {
      setCurrentRoute(resolveInitialRoute());
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  return (
    <NavigationContext.Provider value={{ currentRoute, navigate }}>
      {children}
    </NavigationContext.Provider>
  );
}

export function useNavigation(): NavigationContextValue {
  return useContext(NavigationContext);
}
