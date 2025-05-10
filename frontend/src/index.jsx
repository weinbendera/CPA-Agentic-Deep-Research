import React from 'react';
import ReactDOM from 'react-dom/client';

import App from './components/App.jsx';

/**
 * Initialize the app and render the application
 * Document Object Model
 */
ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
);