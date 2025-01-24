#!/usr/bin/env python3

"""
For debug purposes only. Change API_URL v2 to v1 to POST without authentication
"""

import datetime
import json
import os
import pprint
from pathlib import Path

import requests

API_URL = "https://8np7tbux24.execute-api.us-east-1.amazonaws.com/v2/log"

def parse_iso8601(s):
    return datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S%z')

def get_duration(obj):
    if obj.get('started_at') and obj.get('completed_at'):
        return parse_iso8601(obj['completed_at']) - parse_iso8601(obj['started_at'])

def get_duration_s(obj):
    dur = get_duration(obj)
    return dur.seconds if dur else None

def parse_step(step: dict):
    return {
        'number':     step['number'],
        'name':       step['name'],
        'conclusion': step['conclusion'],
        'duration_s': get_duration_s(step),
    }

def parse_job(job: dict):
    job_id = job['id']
    stats_filename = Path(f"RunnerSysStats-{job_id}.json")
    if not stats_filename.exists():
        return None

    runner_stats = None
    try:
        with open(stats_filename, 'r') as f:
            runner_stats = json.load(f)
        return {
            'id':                job['id'],
            'conclusion':        job['conclusion'],
            'duration_s':        get_duration_s(job),
            'steps':             [parse_step(step) for step in job['steps']],
            'target_os':         runner_stats['target_os'],
            'target_arch':       runner_stats['target_arch'],
            'compiler':          runner_stats['compiler'],
            'build_config':      runner_stats['build_config'],
            'runner_name':       job['runner_name'],
            'runner_group_name': job['runner_group_name'],
            'runner_cpu_count':  runner_stats['cpu_count'],
            'runner_ram_mb':     runner_stats['ram_mb'],
        }
    except:
        print("Something went wrong while parsing the job/runner info. Debug info:")
        pprint.pp(job)
        pprint.pp(runner_stats)
        # re-throw the exception
        raise

def parse_jobs(jobs: list[dict]):
    return [
        job
        for job in [
            parse_job(job)
            for job in jobs
        ]
        if job is not None
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

    resp = requests.post(API_URL, json=result, headers={
        'Authorization': f'Bearer {os.environ.get("CI_STATS_AUTH_TOKEN")}',
    })
    if resp.status_code != 200:
        raise RuntimeError(f'{resp.status_code}: {resp.text}')