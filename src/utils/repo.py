import os
from typing import Tuple

from git import Repo


def get_repo_info() -> Tuple[str, str]:
    repo = Repo(os.path.dirname(os.path.realpath(__file__)), search_parent_directories=True)
    current_commit = repo.head.commit.hexsha

    # Find current tag
    map_tag_commit = {repo.commit(t): t for t in repo.tags}

    current_tag = ""
    for p in repo.iter_commits():
        if p in map_tag_commit:
            current_tag = str(map_tag_commit[p])

    return current_commit, current_tag
