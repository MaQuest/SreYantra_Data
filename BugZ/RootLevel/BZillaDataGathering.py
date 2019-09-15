import bugsy

bugzilla = bugsy.Bugsy(username="gouri.ginde@gmail.com", api_key="c3pW7PACOXcRA7MCUAVAvAldwbszSoD3LjFUbrji",
bugzilla_url="https://bugzilla.mozilla.org/rest", password="lunatic123")
bug = bugzilla.get(1572368)
print (bug.summary)
print (type(bug))
#x = bug.to_dict
print (bug.status, bug.product, bug.version,bug.to_dict()) # bug.depends_on.count)

