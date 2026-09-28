import argparse
from pathlib import Path

from bot import BookBot


def main():
    ap = argparse.ArgumentParser(prog="bookbot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="ingest PDFs")
    p_add.add_argument("paths", nargs="+")

    sub.add_parser("list", help="list ingested books")

    p_ask = sub.add_parser("ask", help="query the library")
    p_ask.add_argument("query")
    p_ask.add_argument("--book", help="restrict to a book id")
    p_ask.add_argument("--top-k", type=int, default=5)

    sub.add_parser("chat", help="interactive loop")

    args = ap.parse_args()
    bot = BookBot()

    if args.cmd == "add":
        for p in args.paths:
            path = Path(p)
            if path.is_dir():
                for pdf in sorted(path.rglob("*.pdf")):
                    bot.add_pdf(pdf)
            elif path.suffix.lower() == ".pdf":
                bot.add_pdf(path)
            else:
                print(f"[skip] not a PDF: {p}")

    elif args.cmd == "list":
        books = bot.store.list_books()
        if not books:
            print("(library is empty)")
        for b in books:
            print(f"{b['id']:<40} {b['title']}  —  {b['pages']}p / {b['chunks']} chunks")

    elif args.cmd == "ask":
        print(bot.ask(args.query, top_k=args.top_k, book_id=args.book))

    elif args.cmd == "chat":
        print("BookBot — offline, no filters, semantic search. Type 'exit' to quit.")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q or q.lower() in ("exit", "quit"):
                break
            print()
            print(bot.ask(q))


if __name__ == "__main__":
    main()