const { spawn } = require('child_process');
const path = require('path');

console.log('🎨 Starting AI Creative Platform...\n');

// Start backend
console.log('🐍 Starting FastAPI backend...');
const backend = spawn('python', ['-m', 'uvicorn', 'main:app', '--reload', '--host', '0.0.0.0', '--port', '8000'], {
    cwd: path.join(__dirname, 'backend'),
    stdio: 'inherit',
    shell: true
});

// Start frontend after a short delay
setTimeout(() => {
    console.log('⚛️ Starting React frontend...');
    const frontend = spawn('npm', ['run', 'dev'], {
        cwd: path.join(__dirname, 'frontend'),
        stdio: 'inherit',
        shell: true
    });

    frontend.on('close', (code) => {
        console.log(`Frontend process exited with code ${code}`);
        backend.kill();
    });
}, 2000);

backend.on('close', (code) => {
    console.log(`Backend process exited with code ${code}`);
    process.exit(code);
});

// Handle Ctrl+C
process.on('SIGINT', () => {
    console.log('\n🛑 Shutting down servers...');
    backend.kill();
    process.exit(0);
});