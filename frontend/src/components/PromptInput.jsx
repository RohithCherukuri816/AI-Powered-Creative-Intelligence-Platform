import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Send, Palette } from 'lucide-react';

const PromptInput = ({ onGenerate, isLoading, inputStyle, buttonStyle, suggestionStyle, recognizedLabel, confidence }) => {
    const [prompt, setPrompt] = useState('');

    const suggestions = [
        "watercolor painting style",
        "minimalist line art",
        "vibrant digital art",
        "vintage poster design",
        "modern geometric style",
        "hand-drawn illustration",
        "ink sketch with splatters",
        "pop art vibrant colors"
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

    // Default styles if no props provided
    const defaultInputStyle = {
        background: 'linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%)',
        border: '3px solid #fdba74',
        borderRadius: '20px',
        boxShadow: '0 15px 35px rgba(251, 146, 60, 0.12)'
    };

    const defaultButtonStyle = {
        background: 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)',
        hoverBackground: 'linear-gradient(135deg, #ea580c 0%, #c2410c 100%)',
        shadow: '0 10px 25px rgba(234, 88, 12, 0.3)'
    };

    const defaultSuggestionStyle = {
        background: 'rgba(254, 215, 170, 0.4)',
        hoverBackground: 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)',
        border: '2px solid #fdba74',
        color: '#7c2d12'
    };

    // Merge default styles with props
    const mergedInputStyle = { ...defaultInputStyle, ...inputStyle };
    const mergedButtonStyle = { ...defaultButtonStyle, ...buttonStyle };
    const mergedSuggestionStyle = { ...defaultSuggestionStyle, ...suggestionStyle };

    return (
        <motion.div
            className="rounded-2xl p-6 max-w-4xl mx-auto border border-orange-200/50 shadow-xl"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            style={{
                background: 'linear-gradient(135deg, rgba(255, 247, 237, 0.8) 0%, rgba(255, 237, 213, 0.9) 100%)',
                backdropFilter: 'blur(10px)'
            }}
        >
            {/* Header */}
            <div className="text-center mb-8">
                <motion.div
                    className="inline-flex items-center gap-3 mb-4 p-3 bg-white/80 rounded-2xl border border-orange-200 shadow-lg"
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.2 }}
                >
                    <div className="w-12 h-12 bg-gradient-to-br from-orange-400 to-rose-500 rounded-xl flex items-center justify-center shadow-lg">
                        <Palette className="text-white" size={24} />
                    </div>
                    <div className="text-left">
                        <h2 className="text-xl font-bold text-gray-800 font-serif">
                            Describe Your Vision
                        </h2>
                        <p className="text-sm text-gray-600">
                            How should we style your creation?
                        </p>
                    </div>
                </motion.div>
            </div>

            {/* Input Form */}
            <form onSubmit={handleSubmit} className="space-y-6">
                <div className="relative">
                    <motion.textarea
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="Describe the style you want... e.g., 'Turn this into a vibrant watercolor painting with soft pastel colors and dreamy atmosphere'"
                        className="w-full p-5 pr-20 rounded-xl resize-none h-28 placeholder-orange-400/60 focus:outline-none transition-all duration-300 backdrop-blur-sm font-medium text-gray-700"
                        disabled={isLoading}
                        style={mergedInputStyle}
                        whileFocus={{ scale: 1.01 }}
                    />

                    {/* Character counter */}
                    <div className="absolute bottom-3 left-5">
                        <motion.span
                            className={`text-xs font-medium ${prompt.length > 100 ? 'text-rose-500' : 'text-orange-500/70'
                                }`}
                            animate={{ scale: prompt.length > 100 ? [1, 1.1, 1] : 1 }}
                            transition={{ duration: 0.3 }}
                        >
                            {prompt.length}/150
                        </motion.span>
                    </div>

                    {/* Submit Button */}
                    <motion.button
                        type="submit"
                        disabled={!prompt.trim() || isLoading}
                        className="absolute right-4 top-4 p-3 text-white rounded-xl disabled:opacity-50 disabled:cursor-not-allowed shadow-lg transition-all duration-300 flex items-center justify-center"
                        style={{
                            background: mergedButtonStyle.background,
                            boxShadow: mergedButtonStyle.shadow
                        }}
                        whileHover={{
                            scale: isLoading ? 1 : 1.1
                        }}
                        whileTap={{ scale: 0.9 }}
                    >
                        {isLoading ? (
                            <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                            />
                        ) : (
                            <motion.div
                                initial={{ scale: 0.8 }}
                                animate={{ scale: 1 }}
                                className="flex items-center gap-1"
                            >
                                <Send size={18} />
                                <span className="text-xs font-bold">GO</span>
                            </motion.div>
                        )}
                    </motion.button>
                </div>

                {/* Main Generate Button */}
                <motion.button
                    type="submit"
                    disabled={!prompt.trim() || isLoading}
                    className="w-full py-4 text-white font-bold rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-lg text-lg relative overflow-hidden group"
                    style={{
                        background: mergedButtonStyle.background,
                        boxShadow: mergedButtonStyle.shadow
                    }}
                    whileHover={{
                        scale: isLoading ? 1 : 1.02
                    }}
                    whileTap={{ scale: 0.98 }}
                >
                    {/* Animated background */}
                    <motion.div
                        className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
                        animate={{ x: [-100, 100] }}
                        transition={{ duration: 1.5, repeat: Infinity, repeatDelay: 2 }}
                    />

                    {isLoading ? (
                        <motion.span
                            className="flex items-center justify-center gap-3 relative z-10"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                        >
                            <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                className="w-6 h-6 border-2 border-white border-t-transparent rounded-full"
                            />
                            Crafting Your Masterpiece...
                        </motion.span>
                    ) : (
                        <motion.span
                            className="flex items-center justify-center gap-3 relative z-10"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                        >
                            <Sparkles size={22} className="text-amber-200" />
                            <span className="bg-gradient-to-r from-amber-200 to-yellow-200 bg-clip-text text-transparent">
                                Create Magic
                            </span>
                            <Sparkles size={22} className="text-amber-200" />
                        </motion.span>
                    )}
                </motion.button>
            </form>

            {/* Recognition-Based Suggestions */}
            {recognizedLabel && (
                <motion.div
                    className="mt-6 p-4 bg-blue-50/80 border border-blue-200 rounded-xl"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                >
                    <div className="text-center mb-3">
                        <p className="text-sm font-medium text-blue-700 inline-flex items-center gap-2">
                            🎯 Detected: <span className="capitalize font-bold">{recognizedLabel.replace(/-/g, ' ')}</span>
                            {confidence && (
                                <span className="text-xs bg-blue-100 px-2 py-1 rounded-full">
                                    {Math.round(confidence * 100)}%
                                </span>
                            )}
                        </p>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                        {[
                            `realistic ${recognizedLabel.replace(/-/g, ' ')}`,
                            `watercolor ${recognizedLabel.replace(/-/g, ' ')}`,
                            `cartoon ${recognizedLabel.replace(/-/g, ' ')}`,
                            `vintage ${recognizedLabel.replace(/-/g, ' ')}`
                        ].map((suggestion, index) => (
                            <motion.button
                                key={index}
                                onClick={() => handleSuggestionClick(suggestion)}
                                className="px-3 py-2 text-xs bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg transition-colors duration-200 font-medium"
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                            >
                                {suggestion}
                            </motion.button>
                        ))}
                    </div>
                </motion.div>
            )}

            {/* Style Suggestions */}
            <motion.div
                className="mt-8"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.6 }}
            >
                <div className="text-center mb-4">
                    <motion.p
                        className="text-sm font-medium text-gray-600 inline-flex items-center gap-2 bg-white/50 px-4 py-2 rounded-full border border-orange-200"
                        whileHover={{ scale: 1.05 }}
                    >
                        <Sparkles size={14} className="text-orange-500" />
                        Quick style ideas
                        <Sparkles size={14} className="text-orange-500" />
                    </motion.p>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {suggestions.map((suggestion, index) => (
                        <motion.button
                            key={index}
                            onClick={() => handleSuggestionClick(suggestion)}
                            className="px-4 py-3 text-sm rounded-xl border-2 font-medium transition-all duration-300 text-center group relative overflow-hidden"
                            style={{
                                background: mergedSuggestionStyle.background,
                                border: mergedSuggestionStyle.border,
                                color: mergedSuggestionStyle.color
                            }}
                            whileHover={{
                                scale: 1.05
                            }}
                            whileTap={{ scale: 0.95 }}
                            disabled={isLoading}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.1 * index }}
                        >
                            {/* Hover effect overlay */}
                            <motion.div
                                className="absolute inset-0 bg-gradient-to-br from-orange-500/0 to-rose-500/0 group-hover:from-orange-500 group-hover:to-rose-500"
                                initial={false}
                                whileHover={{ scale: 1.1 }}
                            />
                            <span className="relative z-10 font-semibold">
                                {suggestion}
                            </span>
                        </motion.button>
                    ))}
                </div>
            </motion.div>

            {/* Tips */}
            <motion.div
                className="mt-6 p-4 bg-white/50 rounded-xl border border-orange-200/50"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.8 }}
            >
                <div className="flex items-start gap-3">
                    <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                        <Sparkles size={16} className="text-orange-600" />
                    </div>
                    <div>
                        <p className="text-sm font-medium text-gray-700 mb-1">Pro Tip</p>
                        <p className="text-xs text-gray-600">
                            Be specific! Include details about colors, textures, art style, and mood for the best results.
                        </p>
                    </div>
                </div>
            </motion.div>
        </motion.div>
    );
};

export default PromptInput;