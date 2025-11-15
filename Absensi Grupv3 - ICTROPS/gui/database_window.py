import customtkinter as ctk
from tkinter import messagebox

class DatabaseWindow:
    def __init__(self, parent, db_manager):
        self.parent = parent
        self.db_manager = db_manager
        self.setup_window()
    
    def setup_window(self):
        """Setup window manajemen database"""
        self.window = ctk.CTkToplevel(self.parent)
        self.window.title("Database Wajah Terdaftar")
        self.window.geometry("500x600")
        self.window.transient(self.parent)
        self.window.grab_set()
        
        # Handle window close event
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

        title_label = ctk.CTkLabel(
            self.window, 
            text="Data Terdaftar", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=10)

        self.scroll_frame = ctk.CTkScrollableFrame(self.window)
        self.scroll_frame.pack(pady=10, padx=20, fill="both", expand=True)

        self.refresh_data()

    def on_close(self):
        """Handle ketika window ditutup"""
        if hasattr(self.parent, 'db_window'):
            self.parent.db_window = None
        self.window.destroy()

    def refresh_data(self):
        """Refresh tampilan data"""
        # Clear existing widgets
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        if not self.db_manager.embedding_db:
            empty_label = ctk.CTkLabel(
                self.scroll_frame, 
                text="Database masih kosong.", 
                font=ctk.CTkFont(size=14)
            )
            empty_label.pack(pady=20)
            return

        # Display each user in database
        for user_id, data in self.db_manager.get_all_users():
            self.create_user_entry(user_id, data)

    def create_user_entry(self, user_id, data):
        """Create UI entry for each user"""
        nama = data.get('name', 'N/A')
        
        # Count embeddings (new and old format)
        num_embeddings = 0
        if 'embeddings' in data and isinstance(data['embeddings'], list):
            num_embeddings = len(data['embeddings'])
        elif 'embedding' in data:
            num_embeddings = 1

        entry_frame = ctk.CTkFrame(
            self.scroll_frame, 
            border_width=1, 
            border_color="gray40"
        )
        entry_frame.pack(pady=5, padx=5, fill="x")

        info_text = f"Nama: {nama}\nID Siswa: {user_id}\nJumlah Foto: {num_embeddings}"
        info_label = ctk.CTkLabel(entry_frame, text=info_text, justify="left")
        info_label.pack(side="left", padx=10, pady=10)

        delete_button = ctk.CTkButton(
            entry_frame, 
            text="Hapus", 
            fg_color="#D32F2F", 
            hover_color="#B71C1C", 
            width=80,
            command=lambda uid=user_id: self.hapus_user(uid)
        )
        delete_button.pack(side="right", padx=10, pady=10)

    def hapus_user(self, user_id):
        """Delete user from database"""
        confirm = messagebox.askyesno(
            "Konfirmasi Hapus", 
            f"Anda yakin ingin menghapus data untuk ID: {user_id}?", 
            parent=self.window
        )
        
        if not confirm:
            return

        success = self.db_manager.delete_user(user_id)
        if success:
            self.db_manager.save_database()
            self.parent.status_label.configure(
                text=f"User ID: {user_id} berhasil dihapus.", 
                text_color="lightgreen"
            )
            self.refresh_data()
        else:
            self.parent.status_label.configure(
                text=f"Gagal menghapus: User ID {user_id} tidak ditemukan.", 
                text_color="orange"
            )

    # Method untuk pengecekan window
    def winfo_viewable(self):
        """Check if window is viewable"""
        try:
            return self.window.winfo_viewable()
        except:
            return False

    def winfo_exists(self):
        """Check if window exists"""
        try:
            return self.window.winfo_exists()
        except:
            return False

    def focus(self):
        """Focus ke window"""
        try:
            self.window.focus()
        except:
            pass

    def destroy(self):
        """Destroy window"""
        try:
            self.window.destroy()
        except:
            pass