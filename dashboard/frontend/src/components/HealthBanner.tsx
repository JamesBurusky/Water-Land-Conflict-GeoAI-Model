import { useHealth } from "../hooks/useApiData";

export function HealthBanner() {
  const { data, isLoading, isError } = useHealth();

  if (isLoading) return null;
  if (isError) {
    return (
      <div className="health-banner health-banner-error">
        Cannot reach the backend API. Check that it's running and
        VITE_API_BASE_URL is set correctly.
      </div>
    );
  }
  if (!data) return null;

  const missing = Object.entries(data.available).filter(([, ok]) => !ok).map(([name]) => name);
  if (missing.length === 0) return null;

  return (
    <div className="health-banner">
      Some pipeline outputs aren't available yet, so those panels are hidden: {missing.join(", ")}.
    </div>
  );
}
