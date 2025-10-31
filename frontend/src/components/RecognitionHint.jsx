import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Eye, Lightbulb, Zap } from 'lucide-react';

const RecognitionHint = ({ hasDrawing }) => {
    const [currentHint, setCurrentHint] = useState(null);
    const [showHint, setShowHint] = useState(false);

    // Simple shape-based hints (this would be replaced with actual AI recognition in production)
    const hints = [
        { shape: 'circular', suggestion: 'Circle detected - try drawing a face, sun, or ball!' },
        { shape: 'rectangular', suggestion: 'Rectangle detected - maybe a house, book, or phone?' },
        { shape: 'linear', suggestion: 'Lines detected - could be a tree, stick figure, or building?' },
        { shape: 'complex', suggestion: 'Complex shape - let the AI surprise you!' }
    ];

    useEffect(() => {
        if (hasDrawing) {
            // Simulate shape detection with a random hint
            const randomHint = hints[Math.floor(Math.random() * hints.length)];
            setCurrentHint(randomHint);
            setShowHint(true);

            // Auto-hide after 3 seconds
            const timer = setTimeout(() => {
                setShowHint(false);
            }, 3000);

            return () => clearTimeout(timer);
        } else {
            setShowHint(false);
            setCurrentHint(null);
        }
    }, [hasDrawing]);

    return (
        <AnimatePresence>
            {showHint && currentHint && (
                <motion.div
                    className="absolute top-4 right-4 max-w-xs z-10"
                    initial={{ opacity: 0, x: 20, scale: 0.9 }}
                    animate={{ opacity: 1, x: 0, scale: 1 }}
                    exit={{ opacity: 0, x: 20, scale: 0.9 }}
                    transition={{ duration: 0.3 }}
                >
                    <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-3 rounded-lg shadow-lg border border-white/20">
                        <div className="flex items-start gap-2">
                            <div className="w-6 h-6 bg-white/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                                <Eye size={14} />
                            </div>
                            <div>
                                <div className="text-xs font-medium mb-1 opacity-90">
                                    AI Recognition Hint
                                </div>
                                <div className="text-sm">
                                    {currentHint.suggestion}
                                </div>
                            </div>
                        </div>

                        {/* Animated indicator */}
                        <div className="flex items-center gap-1 mt-2 text-xs opacity-75">
                            <Zap size={12} />
                            <span>Real-time analysis</span>
                        </div>
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};

export default RecognitionHint;