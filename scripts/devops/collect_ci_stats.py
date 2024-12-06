#!/usr/bin/env python3
import datetime
import json
import os
import pprint
from pathlib import Path

import requests

KNOWN_JOBS = [
    'emscripten-build-test / emscripten-build',
    'fedora-build-test / fedora-build-test',
    'macos-build-test / macos-build-test',
    'windows-build-test / windows-build-test',
    'ubuntu-arm64-build-test / ubuntu-arm-build-test',
    'ubuntu-x64-build-test / ubuntu-x64-build-test',
]

def parse_iso8601(s):
    return datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S%z')

def get_duration(obj):
    if 'started_at' in obj and 'completed_at' in obj:
        return parse_iso8601(obj['completed_at']) - parse_iso8601(obj['started_at'])

def parse_step(step: dict):
    return {
        'number':     step['number'],
        'name':       step['name'],
        'conclusion': step['conclusion'],
        'duration_s': get_duration(step).seconds if step['conclusion'] else None,
    }

def parse_job(job: dict):
    job_id = job['id']
    runner_stats = {}
    stats_filename = Path(f"RunnerSysStats-{job_id}.json")
    if stats_filename.is_file():
        with open(stats_filename, 'r') as f:
            runner_stats = json.load(f)

    return {
        'id':                job['id'],
        'conclusion':        job['conclusion'],
        'duration_s':        get_duration(job).seconds if job['conclusion'] else None,
        'steps':             [parse_step(step) for step in job['steps']],
        'target_os':         runner_stats.get('target_os'),
        'target_arch':       runner_stats.get('target_arch'),
        'compiler':          runner_stats.get('compiler'),
        'build_config':      runner_stats.get('build_config'),
        'runner_name':       job['runner_name'],
        'runner_group_name': job['runner_group_name'],
        'runner_cpu_count':  runner_stats.get('cpu_count'),
        'runner_ram_mb':     runner_stats.get('ram_mb'),
    }

def parse_jobs(jobs: list[dict]):
    return [
        parse_job(job)
        for job in jobs
        if any(
            job['name'].startswith(job_prefix)
            for job_prefix in KNOWN_JOBS
        )
    ]

def fetch_jobs(repo: str, run_id: str):
    return requests.get(f'https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs', headers={
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'Bearer {os.environ.get("GITHUB_TOKEN")}',
        'X-GitHub-Api-Version': '2022-11-28',
    })

if __name__ == "__main__":
    branch = os.environ.get('GIT_BRANCH')
    commit = os.environ.get('GIT_COMMIT')
    repo = os.environ.get("GITHUB_REPOSITORY")
    ref = os.environ.get("GITHUB_REF")
    run_id = os.environ.get("GITHUB_RUN_ID")

    resp = fetch_jobs(repo, run_id)

    result = {
        'id':          int(run_id),
        'git_commit':  commit,
        'git_branch':  branch,
        'github_ref':  ref,
        'github_repo': repo,
        'jobs':        parse_jobs(resp.json()['jobs']),
    }
    pprint.pp(result, indent=2, width=150)

    resp = requests.post("https://api.meshinspector.com/ci-stats/v1/log", json=result, headers={
        'Authorization': f'Bearer {os.environ.get("CI_STATS_AUTH_TOKEN")}',
    })
    if resp.status_code != 200:
        raise RuntimeError(f'{resp.status_code}: {resp.text}')
