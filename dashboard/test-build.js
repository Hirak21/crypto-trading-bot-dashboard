const { execSync } = require('child_process');
const path = require('path');

console.log('Testing React dashboard build...');

try {
  // Change to dashboard directory
  process.chdir(path.join(__dirname));
  
  console.log('Installing dependencies...');
  execSync('npm install', { stdio: 'inherit' });
  
  console.log('Running TypeScript check...');
  execSync('npx tsc --noEmit', { stdio: 'inherit' });
  
  console.log('Building project...');
  execSync('npm run build', { stdio: 'inherit' });
  
  console.log('✅ Build successful!');
} catch (error) {
  console.error('❌ Build failed:', error.message);
  process.exit(1);
}