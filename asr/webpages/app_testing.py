import traceback
from pathlib import Path


def run_testing(app, projects):
    with app.test_client() as c:
        for name in projects:
            print(f'Testing {name}')
            c.get(f'/{name}/').data.decode()
            project = projects[name]
            db = project.database
            uid_key = project.uid_key
            n = len(db)
            uids = []
            for row in db.select(include_data=False):
                uids.append(row.get(uid_key))
                if len(uids) == n:
                    break
            print(len(uids))

            for i, uid in enumerate(uids):
                url = f'/{name}/row/{uid}'
                print(f'\rRows: {i + 1}/{len(uids)} {url}',
                      end='', flush=True)
                try:
                    c.get(url).data.decode()
                except KeyboardInterrupt:
                    raise
                except Exception:
                    print()
                    row = db.get(uid=uid)
                    exc = traceback.format_exc()
                    exc += (f'Problem with {uid}: '
                            f'Formula={row.formula} '
                            f'Crystal type={row.crystal_type}\n'
                            + '-' * 20 + '\n')
                    with Path('errors.txt').open(mode='a') as fid:
                        fid.write(exc)
                        print(exc)
