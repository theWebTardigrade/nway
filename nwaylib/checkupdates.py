from __future__ import division, print_function


def checkupdates(current_version=None):
	"""
	Check for program updates on PyPI

	If you want to disable this, create a file
	I_will_check_for_NWAY_updates_myself_thank_you
	in your working directory.
	"""
	try:
		import os
		if os.path.exists('I_will_check_for_NWAY_updates_myself_thank_you'):
			return
		if current_version is None:
			current_version = '4.7.0'
		import json

		import certifi
		import urllib3
		from StringIO import StringIO
		http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
		r = http.request('GET', 'https://pypi.python.org/pypi/nway/json')
		j = json.load(StringIO(r.data))
		version = j['info']['version']
		if current_version != version:
			print()
			print('A newer version of NWAY seems to be available.')
			print('Please update because it may fix bugs.')
			print()
	except Exception:
		pass
	except ImportError:
		pass
	except IOError:
		pass
