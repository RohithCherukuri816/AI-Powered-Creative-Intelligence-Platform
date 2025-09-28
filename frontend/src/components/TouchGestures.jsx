import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Smartphone, Hand, Info } from 'lucide-react';

const TouchGestures = () => {
    const [showMobileHelp, setShowMobileHelp] = useState(false);
    const [isMobile, setIsMobile] = useState(false);

    useEffect(() => {
        // Detect if user is on mobile
        const checkMobile = () => {
            const isMobileDevice = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
            const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
            setIsMobile(isMobileDevice || isTouchDevice);
        };

        checkMobile();
        window.addEventListener('resize', checkMobile);
        return () => window.removeEventListener('resize', checkMobile);
    }, []);

    const gestures = [
        {
            icon: Hand,
            title: 'Draw',
            description: 'Touch and drag to draw on the canvas',
            gesture: 'Single finger drag'
        },

    ];

    if (!isMobile) return null;

    return (
        <>
            {/* Mobile Help Button */}
            <motion.button
                onClick={() => setShowMobileHelp(true)}
                className="fixed bottom-20 left-6 z-40 bg-white/90 backdrop-blur-sm border border-orange-200 text-gray-700 p-3 rounded-full shadow-lg hover:shadow-xl transition-all duration-300 group md:hidden"
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                title="Touch Gestures"
            >
                <Smartphone size={20} className="group-hover:text-orange-600 transition-colors" />
            </motion.button>

            {/* Touch Gestures Modal */}
            <AnimatePresence>
                {showMobileHelp && (
                    <motion.div
                        className="fixed inset-0 z-50 flex items-center justify-center p-4"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                    >
                        {/* Backdrop */}
                        <motion.div
                            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={() => setShowMobileHelp(false)}
                        />

                        {/* Modal */}
                        <motion.div
                            className="relative bg-white rounded-2xl shadow-2xl max-w-md w-full max-h-[80vh] overflow-hidden"
                            initial={{ opacity: 0, scale: 0.9, y: 20 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.9, y: 20 }}
                            transition={{ type: "spring", duration: 0.3 }}
                        >
                            {/* Header */}
                            <div className="bg-gradient-to-r from-orange-500 to-rose-500 text-white p-6">
                                <div className="flex items-center gap-3">
                                    <Smartphone size={24} />
                                    <h2 className="text-xl font-bold">Touch Gestures</h2>
                                </div>
                                <p className="text-white/90 text-sm mt-2">
                                    Learn how to use SketchCraft on your mobile device
                                </p>
                            </div>

                            {/* Content */}
                            <div className="p-6 space-y-6">
                                {gestures.map((gesture, index) => {
                                    const IconComponent = gesture.icon;
                                    return (
                                        <motion.div
                                            key={index}
                                            className="flex items-start gap-4 p-4 bg-gray-50 rounded-xl"
                                            initial={{ opacity: 0, x: -20 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ delay: index * 0.1 }}
                                        >
                                            <div className="w-12 h-12 bg-gradient-to-br from-orange-400 to-rose-500 rounded-xl flex items-center justify-center flex-shrink-0">
                                                <IconComponent size={20} className="text-white" />
                                            </div>
                                            <div className="flex-1">
                                                <h3 className="font-semibold text-gray-800 mb-1">
                                                    {gesture.title}
                                                </h3>
                                                <p className="text-sm text-gray-600 mb-2">
                                                    {gesture.description}
                                                </p>
                                                <div className="inline-flex items-center gap-2 bg-white px-3 py-1 rounded-full border border-gray-200">
                                                    <Hand size={12} className="text-orange-500" />
                                                    <span className="text-xs font-medium text-gray-700">
                                                        {gesture.gesture}
                                                    </span>
                                                </div>
                                            </div>
                                        </motion.div>
                                    );
                                })}

                                {/* Additional Tips */}
                                <motion.div
                                    className="p-4 bg-blue-50 rounded-xl border border-blue-200"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    transition={{ delay: 0.4 }}
                                >
                                    <div className="flex items-start gap-3">
                                        <Info size={16} className="text-blue-600 flex-shrink-0 mt-0.5" />
                                        <div>
                                            <h4 className="font-medium text-blue-800 text-sm mb-1">
                                                Mobile Tips
                                            </h4>
                                            <ul className="text-xs text-blue-700 space-y-1">
                                                <li>• Use landscape mode for better drawing experience</li>
                                                <li>• Adjust brush size for your finger thickness</li>
                                                <li>• Draw with smooth, steady movements</li>
                                                <li>• Use the eraser tool to make corrections</li>
                                            </ul>
                                        </div>
                                    </div>
                                </motion.div>

                                {/* Close Button */}
                                <motion.button
                                    onClick={() => setShowMobileHelp(false)}
                                    className="w-full bg-gradient-to-r from-orange-500 to-rose-500 text-white font-semibold py-3 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300"
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                >
                                    Got it!
                                </motion.button>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
};

export default TouchGestures;