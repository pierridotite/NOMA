import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    console.log('NOMA Language extension is now active!');

    // Register a command to run the current NOMA file
    const runCommand = vscode.commands.registerCommand('noma.run', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        const document = editor.document;
        if (document.languageId !== 'noma') {
            vscode.window.showErrorMessage('Not a NOMA file');
            return;
        }

        // Save the document first
        await document.save();

        const filePath = document.fileName;
        
        // Create or get terminal
        let terminal = vscode.window.terminals.find(t => t.name === 'NOMA');
        if (!terminal) {
            terminal = vscode.window.createTerminal('NOMA');
        }
        
        terminal.show();
        terminal.sendText(`cargo run -- run "${filePath}"`);
    });

    // Register a command to build the current NOMA file
    const buildCommand = vscode.commands.registerCommand('noma.build', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        const document = editor.document;
        if (document.languageId !== 'noma') {
            vscode.window.showErrorMessage('Not a NOMA file');
            return;
        }

        await document.save();

        const filePath = document.fileName;
        
        let terminal = vscode.window.terminals.find(t => t.name === 'NOMA');
        if (!terminal) {
            terminal = vscode.window.createTerminal('NOMA');
        }
        
        terminal.show();
        terminal.sendText(`cargo run -- build "${filePath}"`);
    });

    context.subscriptions.push(runCommand, buildCommand);
}

export function deactivate() {}
