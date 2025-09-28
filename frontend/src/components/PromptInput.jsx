import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Send } from 'lucide-react';

const PromptInput = ({ onGenerate, isLoading }) => {
    const [prompt, setPrompt] = useState('');

    const suggestions = [
        "watercolor painting style",
        "minimalist line art",
        "vibrant digital art",
        "vintage poster design",
        "modern geometric style",
        "hand-drawn illustration"
    ];

    const handleSubmit = (e) => {
        e.preventDefault();
        if (prompt.trim() && !isLoading) {
            onGenerate(prompt.trim());
        }
    };

    const handleSuggestionClick = (suggestion) => {
        setPrompt(suggestion);
    };

    return (
        <motion.div
            className="glass-card rounded-2xl p-6 max-w-4xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
        >
            <div className="text-center mb-6">
                <h2 className="text-2xl font-bold text-gray-800 mb-2 flex items-center justify-center gap-2">
                    <Sparkles className="text-pastel-purple" />
                    Describe Your Style
                </h2>
                <p className="text-gray-600">
                    Tell the AI how you want your doodle transformed
                </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
                <div className="relative">
                    <textarea
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="e.g., Transform this into a beautiful watercolor painting with soft pastels and dreamy clouds..."
                        className="w-full p-4 pr-16 rounded-xl glow-border resize-none h-24 bg-white/90 placeholder-gray-400 focus:outline-none transition-all duration-300"
                        disabled={isLoading}
                    />

                    <motion.button
                        type="submit"
                        disabled={!prompt.trim() || isLoading}
                        className="absolute right-3 top-3 p-3 bg-gradient-to-r from-pastel-purple to-pastel-pink text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transition-all duration-300"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                    >
                        {isLoading ? (
                            <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                            />
                        ) : (
                            <Send size={20} />
                        )}
                    </motion.button>
                </div>

                <motion.button
                    type="submit"
                    disabled={!prompt.trim() || isLoading}
                    className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                >
                    {isLoading ? (
                        <span className="flex items-center justify-center gap-2">
                            <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                            />
                            Generating Magic...
                        </span>
                    ) : (
                        <span className="flex items-center justify-center gap-2">
                            <Sparkles size={20} />
                            Generate Design
                        </span>
                    )}
                </motion.button>
            </form>

            {/* Style Suggestions */}
            <div className="mt-6">
                <p className="text-sm text-gray-600 mb-3 text-center">
                    Or try these popular styles:
                </p>
                <div className="flex flex-wrap gap-2 justify-center">
                    {suggestions.map((suggestion, index) => (
                        <motion.button
                            key={index}
                            onClick={() => handleSuggestionClick(suggestion)}
                            className="px-4 py-2 text-sm bg-white/80 text-pastel-purple rounded-full border border-pastel-purple/30 hover:bg-pastel-purple hover:text-white transition-all duration-300"
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            disabled={isLoading}
                        >
                            {suggestion}
                        </motion.button>
                    ))}
                </div>
            </div>
        </motion.div>
    );
};

export default PromptInput;