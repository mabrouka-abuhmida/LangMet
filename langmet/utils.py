from datetime import datetime, timedelta, timezone
from .models import CitationMessageEvent, CompletionEvent, RagEvent

#1,Walks through dicts, lists, tuples
#2,Looks for keys named "created_at"
#3,If the value is a number (seconds), it converts it back into a datetime
def reconstruct_fromjson(obj):
    #print(CompletionEvent.__dataclass_fields__.keys())#debug use

    # Handle lists
    if isinstance(obj, list):
        return [reconstruct_fromjson(x) for x in obj]

    # Handle tuples
    if isinstance(obj, tuple):
        return tuple(reconstruct_fromjson(x) for x in obj)

    # Handle dicts
    if isinstance(obj, dict):
        # First recursively reconstruct children
        new = {k: reconstruct_fromjson(v) for k, v in obj.items()}

        # Fix created_at if numeric
        if "created_at" in new and isinstance(new["created_at"], (int, float)):
            new["created_at"] = datetime.now(timezone.utc) - timedelta(seconds=new["created_at"])

        # Try reconstructing known dataclasses
        try:
            if set(new.keys()) == set(CompletionEvent.__dataclass_fields__.keys()):
                return CompletionEvent(**new)
            if set(new.keys()) == set(RagEvent.__dataclass_fields__.keys()):
                return RagEvent(**new)
            if set(new.keys()) == set(CitationMessageEvent.__dataclass_fields__.keys()):
                return CitationMessageEvent(**new)
        except Exception:
            print("has error at "+str(new))#debug use,should not reachhere
            pass

        return new

    return obj