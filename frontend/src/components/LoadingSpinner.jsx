import React from 'react';
import { motion } from 'framer-motion';

const LoadingSpinner = ({ message = "Generating your masterpiece..." }) => {
    return (
        <motion.div
            className="flex flex-col items-center justify-center p-8"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
        >
            <div className="relative">
                {/* Outer ring */}
                <motion.div
                    className="w-16 h-16 border-4 border-pastel-purple/30 rounded-full"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                />

                {/* Inner ring */}
                <motion.div
                    className="absolute inset-2 w-12 h-12 border-4 border-pastel-pink border-t-transparent rounded-full"
                    animate={{ rotate: -360 }}
                    transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                />

                {/* Center dot */}
                <motion.div
                    className="absolute inset-6 w-4 h-4 bg-gradient-to-r from-pastel-purple to-pastel-pink rounded-full"
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 1, repeat: Infinity }}
                />
            </div>

            <motion.p
                className="mt-4 text-gray-600 font-medium"
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 2, repeat: Infinity }}
            >
                {message}
            </motion.p>

            <motion.div
                className="flex gap-1 mt-2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
            >
                {[0, 1, 2].map((i) => (
                    <motion.div
                        key={i}
                        className="w-2 h-2 bg-pastel-purple rounded-full"
                        animate={{ y: [0, -8, 0] }}
                        transition={{
                            duration: 0.6,
                            repeat: Infinity,
                            delay: i * 0.2
                        }}
                    />
                ))}
            </motion.div>
        </motion.div>
    );
};

export default LoadingSpinner;