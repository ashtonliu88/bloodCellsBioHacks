module.exports = {
    // Whether to hide emojis
    disableEmoji: false,

    // List of commit types (must match the keys in `types`)
    list: [
        'Add',
        'Update',
        'Test',
        'Fix',
        'Docs',
        'Refactor',
        'Style',
        'CI',
        'Performance',
        'Config',
        'Package',
        'Security',
        'Tracking',
        'Files'
    ],

    // Maximum commit message length
    maxMessageLength: 64,

    // Minimum commit message length
    minMessageLength: 3,

    // Types of questions asked in git-cz
    // questions: ['type', 'scope', 'subject', 'body', 'breaking', 'issues', 'lerna'],
    questions: ['type', 'subject'],

    // Available scopes (categories within the project)
    scopes: ['None', 'API', 'Feature', 'Setup', 'Type Definitions'],

    // Commit types and their descriptions
    types: {
        // 🐞 Bugs and Performance
        Fix: { 
            description: 'Bug fixes', 
            emoji: '🐛', 
            value: 'Fix' 
        },
        CriticalFix: { 
            description: 'Fix a critical bug', 
            emoji: '🚑', 
            value: 'CriticalFix' 
        },
        Performance: { 
            description: 'Performance improvements', 
            emoji: '🚀', 
            value: 'Performance' 
        },

        // 💻 Code Quality and Style
        Improve: { 
            description: 'Feature improvements', 
            emoji: '👍', 
            value: 'Improve' 
        },
        Refactor: { 
            description: 'Code refactoring', 
            emoji: '♻️', 
            value: 'Refactor' 
        },
        Style: { 
            description: 'Linting and code style fixes', 
            emoji: '👕', 
            value: 'Style' 
        },

        // 🎨 UI/UX and Design
        Add: { 
            description: 'Add new features', 
            emoji: '✨', 
            value: 'Add' 
        },
        Design: { 
            description: 'Design changes only', 
            emoji: '🎨', 
            value: 'Design' 
        },

        // 🛠️ Development Tools and Settings
        WIP: { 
            description: 'Work in progress', 
            emoji: '🚧', 
            value: 'WIP' 
        },
        Config: { 
            description: 'Configuration file changes', 
            emoji: '⚙', 
            value: 'Config' 
        },
        Package: { 
            description: 'Add new dependencies', 
            emoji: '📦', 
            value: 'Package' 
        },
        Update: { 
            description: 'Update dependencies', 
            emoji: '🆙', 
            value: 'Update' 
        },

        // 📚 Documentation and Comments
        Wording: { 
            description: 'Fix wording', 
            emoji: '📝', 
            value: 'Wording' 
        },
        Docs: { 
            description: 'Documentation updates', 
            emoji: '📚', 
            value: 'Docs' 
        },
        Comment: { 
            description: 'Add comments or ideas', 
            emoji: '💡', 
            value: 'Comment' 
        },

        // 🛡️ Security
        Security: { 
            description: 'Security-related improvements', 
            emoji: '👮', 
            value: 'Security' 
        },

        // 🧪 Testing and CI
        Test: { 
            description: 'Fix/improve testing & CI', 
            emoji: '💚', 
            value: 'Test' 
        },
        CI: { 
            description: 'CI/CD-related changes', 
            emoji: '🎡', 
            value: 'CI' 
        },

        // 🗂️ File and Folder Manipulation
        Files: { 
            description: 'Folder and file manipulation', 
            emoji: '📂', 
            value: 'Files' 
        },
        Move: { 
            description: 'Move files', 
            emoji: '🚚', 
            value: 'Move' 
        },

        // 📊 Logging and Tracking
        Log: { 
            description: 'Add logging', 
            emoji: '🔊', 
            value: 'Log' 
        },
        LogDelete: { 
            description: 'Delete log entries', 
            emoji: '🔇', 
            value: 'LogDelete' 
        },
        Tracking: { 
            description: 'Add analytics or tracking code', 
            emoji: '📈', 
            value: 'Tracking' 
        },

        // 💡 Other
        Review: { 
            description: 'Code reading and questions', 
            emoji: '🧐', 
            value: 'Review' 
        },
        Fun: { 
            description: 'Fun-to-write code', 
            emoji: '🍻', 
            value: 'Fun' 
        },
        Ignore: { 
            description: 'Add to .gitignore', 
            emoji: '🙈', 
            value: 'Ignore' 
        },
        BasicFix: { 
            description: 'Basic bug fixes and problem-solving', 
            emoji: '🛠️', 
            value: 'BasicFix'
         }
    }
};
