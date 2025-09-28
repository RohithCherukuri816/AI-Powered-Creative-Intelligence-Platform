import React from 'react';
import { motion } from 'framer-motion';
import { Download, RefreshCw, Heart, Share2 } from 'lucide-react';

const GeneratedImageCard = ({ imageUrl, onTryAgain, onDownload }) => {
    if (!imageUrl) return null;

    const handleDownload = () => {
        if (onDownload) {
            onDownload();
        } else {
            // Default download behavior
            const link = document.createElement('a');
            link.href = imageUrl;
            link.download = `ai-generated-design-${Date.now()}.png`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    };

    const handleShare = async () => {
        if (navigator.share) {
            try {
                await navigator.share({
                    title: 'Check out my AI-generated design!',
                    text: 'I created this amazing design using AI-Powered Creative Intelligence Platform',
                    url: window.location.href
                });
            } catch (error) {
                console.log('Error sharing:', error);
            }
        } else {
            // Fallback: copy to clipboard
            navigator.clipboard.writeText(window.location.href);
            // You could show a toast notification here
        }
    };

    return (
        <motion.div
            className="glass-card rounded-2xl p-6 max-w-4xl mx-auto"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, type: "spring", bounce: 0.3 }}
        >
            <div className="text-center mb-6">
                <motion.h2
                    className="text-2xl font-bold text-gray-800 mb-2"
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                >
                    ✨ Your AI-Generated Design
                </motion.h2>
                <motion.p
                    className="text-gray-600"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.3 }}
                >
                    Amazing! Here's your transformed artwork
                </motion.p>
            </div>

            {/* Generated Image */}
            <motion.div
                className="relative mb-6 flex justify-center"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
            >
                <div className="relative group">
                    <motion.img
                        src={imageUrl}
                        alt="AI Generated Design"
                        className="max-w-full h-auto rounded-xl shadow-2xl border-4 border-white/50"
                        whileHover={{ scale: 1.02 }}
                        transition={{ duration: 0.3 }}
                    />

                    {/* Glow effect */}
                    <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-pastel-purple/20 to-pastel-pink/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300 animate-glow" />
                </div>
            </motion.div>

            {/* Action Buttons */}
            <motion.div
                className="flex flex-wrap gap-4 justify-center"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
            >
                <motion.button
                    onClick={handleDownload}
                    className="flex items-center gap-2 bg-gradient-to-r from-green-400 to-green-500 text-white font-semibold py-3 px-6 rounded-full shadow-lg hover:shadow-xl transition-all duration-300"
                    whileHover={{ scale: 1.05, y: -2 }}
                    whileTap={{ scale: 0.95 }}
                >
                    <Download size={20} />
                    Download
                </motion.button>

                <motion.button
                    onClick={onTryAgain}
                    className="flex items-center gap-2 bg-gradient-to-r from-pastel-purple to-pastel-pink text-white font-semibold py-3 px-6 rounded-full shadow-lg hover:shadow-xl transition-all duration-300"
                    whileHover={{ scale: 1.05, y: -2 }}
                    whileTap={{ scale: 0.95 }}
                >
                    <RefreshCw size={20} />
                    Try Again
                </motion.button>

                <motion.button
                    onClick={handleShare}
                    className="flex items-center gap-2 btn-secondary"
                    whileHover={{ scale: 1.05, y: -2 }}
                    whileTap={{ scale: 0.95 }}
                >
                    <Share2 size={18} />
                    Share
                </motion.button>

                <motion.button
                    className="flex items-center gap-2 btn-secondary group"
                    whileHover={{ scale: 1.05, y: -2 }}
                    whileTap={{ scale: 0.95 }}
                >
                    <Heart size={18} className="group-hover:fill-current transition-all" />
                    Save to Favorites
                </motion.button>
            </motion.div>

            {/* Stats or Info */}
            <motion.div
                className="mt-6 p-4 bg-white/50 rounded-xl text-center"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.6 }}
            >
                <p className="text-sm text-gray-600">
                    🎨 Generated in seconds • ✨ Powered by AI • 💜 Made with love
                </p>
            </motion.div>
        </motion.div>
    );
};

export default GeneratedImageCard;