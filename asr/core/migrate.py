def is_migratable(obj):
    if hasattr(obj, 'migrate'):
        if obj.migrate is not None:
            return True
    return False
