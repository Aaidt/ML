import os
import sys
from textual.app import App, ComposeResult
from textual.widgets import Header, Static, RichLog, Input, Button, Label
from textual.containers import Horizontal, Vertical, Container
from textual.reactive import reactive
from textual import work

from rlm_core import read_context, get_context_info, run_rlm, load_api_key


class ContextPanel(Static):
    def update_info(self, info: dict):
        lines = []
        lines.append("[bold]Context Path[/]")
        lines.append(f"  {info['path']}")
        lines.append("")
        lines.append("[bold]Stats[/]")
        lines.append(f"  Size:  {_fmt_size(info['size'])}")
        lines.append(f"  Lines: {info['lines']:,}")
        lines.append(f"  Files: {info['files']}")
        lines.append(f"  Type:  {'Directory' if info['is_dir'] else 'File'}")
        self.update("\n".join(lines))

    def show_no_context(self):
        self.update("[dim]No context loaded.\nSet a file/folder path below.[/]")

    def show_loading(self):
        self.update("[yellow]Loading context...[/]")


class AgentLog(RichLog):
    def write_info(self, msg: str):
        self.write(f"[bold blue]●[/] {msg}")

    def write_code(self, code: str):
        self.write("")
        self.write("[bold yellow]📝 LLM wrote[/]")
        for line in code.strip().split("\n"):
            self.write(f"  [dim]{line}[/]")
        self.write("")

    def write_output(self, output: str):
        self.write("[bold cyan]📤 REPL output[/]")
        for line in output.strip().split("\n"):
            self.write(f"  {line}")
        self.write("")

    def write_iteration(self, i: int, total: int):
        sep = "─" * 40
        self.write(f"[bold green]{sep}[/]")
        self.write(f"[bold green]  Iteration {i}/{total}[/]")
        self.write(f"[bold green]{sep}[/]")

    def write_final(self, answer: str, iterations: int, sub_calls: int):
        self.write("")
        self.write("[bold magenta]══════════════════════════════[/]")
        self.write(f"[bold magenta]  FINAL ANSWER:[/] [bold white]{answer}[/]")
        self.write(
            f"[bold magenta]  Iterations: {iterations}  Sub-LLM calls: {sub_calls}[/]"
        )
        self.write("[bold magenta]══════════════════════════════[/]")
        self.write("")

    def write_error(self, msg: str):
        self.write(f"[bold red]ERROR:[/] {msg}")

    def write_timeout(self, max_iter: int):
        self.write(
            f"[bold red]⚠ Max iterations ({max_iter}) reached without FINAL()[/]"
        )

    def write_welcome(self):
        self.write("[bold green]RLM Agent TUI[/]")
        self.write("[dim]" + "─" * 40 + "[/]")
        self.write("")
        self.write("Set a context path below, then ask a query.")
        self.write("The RLM agent writes Python code to explore")
        self.write("the context and find answers — the full context")
        self.write("never enters the LLM prompt.")
        self.write("")


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class RLMTui(App):
    CSS = """
    #main {
        height: 1fr;
    }

    #sidebar {
        width: 28;
        border: solid $primary;
        padding: 1;
        overflow-y: auto;
    }

    #log {
        border: solid $primary;
        padding: 1;
    }

    .control-row {
        height: 3;
        align: center middle;
    }

    .control-row Label {
        padding: 0 1;
    }

    #path-input {
        width: 1fr;
    }

    #query-input {
        width: 1fr;
    }

    Button {
        margin: 0 1;
    }

    #status-bar {
        height: 1;
        background: $accent;
        color: $text;
        text-align: center;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_log", "Clear Log"),
        ("ctrl+r", "load_context", "Reload Context"),
    ]

    context_data = reactive("")
    context_path = reactive("")

    def compose(self):
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            yield ContextPanel(id="sidebar")
            yield AgentLog(id="log", highlight=True, markup=True)
        with Horizontal(classes="control-row"):
            yield Label("Path:")
            yield Input(placeholder="/path/to/file_or_folder", id="path-input")
            yield Button("Load", id="load-btn", variant="primary")
        with Horizontal(classes="control-row"):
            yield Label("Query:")
            yield Input(
                placeholder="Ask a question about your context...", id="query-input"
            )
            yield Button("Go", id="go-btn", variant="success")
        yield Static(id="status-bar")

    def on_mount(self):
        log = self.query_one(AgentLog)
        log.write_welcome()
        self.query_one(ContextPanel).show_no_context()
        self._update_status("Ready — set a context path and ask a query")

        if not load_api_key():
            log.write_error("OPENROUTER_API_KEY not found in .env or environment!")
            self._update_status("❌ No API key")
        else:
            log.write_info("API key loaded ✓")

        self.query_one("#path-input", Input).focus()

    def _update_status(self, msg: str):
        self.query_one("#status-bar", Static).update(f"  {msg}  ")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "load-btn":
            self._load_context()
        elif event.button.id == "go-btn":
            self._run_query()

    def on_input_submitted(self, event: Input.Submitted):
        if event.input.id == "path-input":
            self._load_context()
        elif event.input.id == "query-input":
            self._run_query()

    def action_clear_log(self):
        log = self.query_one(AgentLog)
        log.clear()
        log.write_welcome()
        self._update_status("Log cleared")

    def action_load_context(self):
        self._load_context()

    def _load_context(self):
        path_input = self.query_one("#path-input", Input)
        path = path_input.value.strip()
        if not path:
            self._update_status("⚠ Enter a path first")
            return

        path = os.path.expanduser(path)
        if not os.path.exists(path):
            self.query_one(AgentLog).write_error(f"Path does not exist: {path}")
            self._update_status("❌ Invalid path")
            return

        panel = self.query_one(ContextPanel)
        panel.show_loading()
        self._update_status(f"Loading context from {path}...")
        self.refresh()

        try:
            info = get_context_info(path)
            content = read_context(path)
            self.context_data = content
            self.context_path = path
            panel.update_info(info)
            log = self.query_one(AgentLog)
            log.write_info(
                f"Context loaded: {info['files']} file(s), {_fmt_size(info['size'])}"
            )
            self._update_status(f"✅ Context loaded: {path}")
        except Exception as e:
            self.query_one(ContextPanel).show_no_context()
            self.query_one(AgentLog).write_error(f"Failed to load context: {e}")
            self._update_status("❌ Load failed")

    def _run_query(self):
        query_input = self.query_one("#query-input", Input)
        query = query_input.value.strip()

        if not query:
            self._update_status("⚠ Enter a query first")
            return

        if not self.context_data:
            self._update_status("⚠ Load a context first")
            self.query_one(AgentLog).write_error(
                "No context loaded. Set a file/folder path and click Load."
            )
            return

        log = self.query_one(AgentLog)
        log.write(f"\n[bold yellow]── Query: {query} ──[/]")
        self._update_status("🤖 RLM agent running...")
        query_input.disabled = True
        self.query_one("#go-btn", Button).disabled = True

        self._run_agent(query)

    @work(thread=True)
    def _run_agent(self, query: str):
        log = self.query_one(AgentLog)
        try:

            def on_iteration(i, total):
                self.call_from_thread(log.write_iteration, i, total)

            def on_code(code):
                self.call_from_thread(log.write_code, code)

            def on_output(output):
                self.call_from_thread(log.write_output, output)

            def on_final(answer, iterations, sub_calls):
                self.call_from_thread(log.write_final, answer, iterations, sub_calls)
                self.call_from_thread(
                    self._update_status,
                    f"🏁 Done — {iterations} iterations, {sub_calls} sub-calls",
                )

            def on_timeout(max_iter):
                self.call_from_thread(log.write_timeout, max_iter)
                self.call_from_thread(self._update_status, "⏱ Timed out")

            callbacks = {
                "on_iteration": on_iteration,
                "on_code": on_code,
                "on_output": on_output,
                "on_final": on_final,
                "on_timeout": on_timeout,
            }

            run_rlm(query, self.context_data, max_iterations=10, callbacks=callbacks)
        except Exception as e:
            self.call_from_thread(log.write_error, str(e))
            self.call_from_thread(self._update_status, f"❌ Error: {e}")
        finally:
            self.call_from_thread(self._reenable_inputs)

    def _reenable_inputs(self):
        qi = self.query_one("#query-input", Input)
        qi.disabled = False
        qi.focus()
        self.query_one("#go-btn", Button).disabled = False


if __name__ == "__main__":
    RLMTui().run()
