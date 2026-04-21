import { useMemo } from "react";
import { useAuth } from "./AuthContext";
import { useNavigation } from "./NavigationContext";
import { AppShell } from "./AppShell";
import { LoginPage } from "../features/auth/LoginPage";
import { LabelingPage } from "../features/labeling/LabelingPage";
import { UploadPage } from "../features/upload/UploadPage";
import { ApiChatService } from "../services/api/chatApi";
import { ApiVideoAnalysisService } from "../services/api/videoAnalysisApi";

export function App(): JSX.Element {
  const { user, loading } = useAuth();
  const { currentRoute } = useNavigation();
  const analysisService = useMemo(() => new ApiVideoAnalysisService(), []);
  const chatService = useMemo(() => new ApiChatService(), []);

  if (loading) {
    return (
      <div className="auth-loading">
        <div className="auth-spinner" />
      </div>
    );
  }

  if (!user) {
    return <LoginPage />;
  }

  return (
    <AppShell>
      {currentRoute === "labeling" ? (
        <LabelingPage />
      ) : (
        <UploadPage analysisService={analysisService} chatService={chatService} />
      )}
    </AppShell>
  );
}
