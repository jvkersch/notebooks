from collections import defaultdict
import random

from starlette.websockets import WebSocket


START_TEXT = "Type any valid word to start the game. The server will respond with a word whose first two letter match the last two letters of your word. You respond in a similar way. The first to run out of options loses."


def get_worddict():
    with open("/usr/share/dict/words") as fp:
        words = [line.strip().lower() for line in fp]
    worddict = defaultdict(list)
    for w in words:
        if len(w) > 2:
            worddict[w[:2]].append(w)
    for c, ws in worddict.items():
        random.shuffle(ws)
        worddict[c] = ws
    return worddict


def make_report(words):
    return ", ".join(f"{c}: {len(ws)}" for c, ws in words.items())
        

async def app(scope, receive, send):
    words = get_worddict()
    all_words = set(sum(words.values(), []))
    
    websocket = WebSocket(scope=scope, receive=receive, send=send)
    await websocket.accept()
    await websocket.send_text(START_TEXT)

    server_word = ""
    while True:
        resp = await websocket.receive_text()
        user_word = resp.lower().strip()

        if user_word == "*":
            report = make_report(words)
            await websocket.send_text(report)
        elif user_word == '!':
            if not server_word:
                hint = "Play a word first."
            else:
                hints = words[server_word[-2:]]
                if len(hints) > 0:
                    hint = hints[0]
                else:
                    hint = "There are no valid options for you."
            await websocket.send_text(hint)
        elif user_word not in all_words:
            await websocket.send_text(f"Not a valid word: {user_word}. You lose.")
            break
        else:
            # move from the user, check and make counter move
            if server_word and server_word[-2:] != user_word[:2]:
                await websocket.send_text(
                    "First two characters don't match. You lose.")
                break
            try:
                server_word = words[user_word[-2:]].pop()
                await websocket.send_text(server_word)
            except (IndexError, KeyError):  # no more guesses, server loses
                await websocket.send_text(
                    "I've run out of options. Good job, you won!")
                break
        
    await websocket.close()
