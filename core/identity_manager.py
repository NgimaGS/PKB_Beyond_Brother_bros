import os

class IdentityManager:
    """Manages the reading and writing of the project's AGENT.md identity file."""
    
    def __init__(self, file_path="AGENT.md"):
        self.file_path = file_path
        self.last_config = self.load_config()

    def load_config(self):
        """Reads AGENT.md and returns a dictionary of the main sections."""
        if not os.path.exists(self.file_path):
            return {"Identity": "N/A", "Persona": "N/A", "Boundaries": "N/A"}
        
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        return content

    def save_config(self, content):
        """Writes the new content to AGENT.md."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception:
            return False

    def is_diff(self, new_content):
        """Checks if the new content is different from the existing file."""
        current = self.load_config()
        return current.strip() != new_content.strip()
