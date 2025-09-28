/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                pastel: {
                    lavender: '#E6E6FA',
                    pink: '#FFB6C1',
                    blue: '#B0E0E6',
                    mint: '#F0FFF0',
                    peach: '#FFDAB9',
                    purple: '#DDA0DD',
                    yellow: '#FFFFE0'
                }
            },
            animation: {
                'float': 'float 6s ease-in-out infinite',
                'glow': 'glow 2s ease-in-out infinite alternate',
                'bounce-slow': 'bounce 3s infinite'
            },
            keyframes: {
                float: {
                    '0%, 100%': { transform: 'translateY(0px)' },
                    '50%': { transform: 'translateY(-20px)' }
                },
                glow: {
                    '0%': { boxShadow: '0 0 5px #E6E6FA, 0 0 10px #E6E6FA, 0 0 15px #E6E6FA' },
                    '100%': { boxShadow: '0 0 10px #DDA0DD, 0 0 20px #DDA0DD, 0 0 30px #DDA0DD' }
                }
            }
        },
    },
    plugins: [],
}