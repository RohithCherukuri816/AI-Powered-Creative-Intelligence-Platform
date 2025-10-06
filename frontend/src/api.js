const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Helper to make absolute URL for static files returned as relative paths
export const toAbsoluteUrl = (pathOrUrl) => {
    if (!pathOrUrl) return pathOrUrl;
    try {
        // If already absolute, return as is
        const u = new URL(pathOrUrl);
        return u.href;
    } catch (e) {
        // Relative path: prefix with backend base
        const base = API_BASE_URL.replace(/\/$/, '');
        const path = pathOrUrl.startsWith('/') ? pathOrUrl : `/${pathOrUrl}`;
        return `${base}${path}`;
    }
};

export const api = {
    async generateDesign(prompt, sketchData) {
        try {
            const response = await fetch(`${API_BASE_URL}/generate-design`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt,
                    sketch: sketchData
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error generating design:', error);
            throw error;
        }
    },

    async healthCheck() {
        try {
            const response = await fetch(`${API_BASE_URL}/health`);
            return await response.json();
        } catch (error) {
            console.error('Health check failed:', error);
            throw error;
        }
    }
};