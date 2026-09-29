from desktop_env.providers.base import VMManager

REMOTE_SENTINEL = "remote://docker_server"


class DockerServerVMManager(VMManager):

    def initialize_registry(self, **kwargs):
        pass

    def add_vm(self, vm_path, **kwargs):
        pass

    def delete_vm(self, vm_path, **kwargs):
        pass

    def occupy_vm(self, vm_path, pid, **kwargs):
        pass

    def list_free_vms(self, **kwargs):
        return REMOTE_SENTINEL

    def check_and_clean(self, **kwargs):
        pass

    def get_vm_path(self, os_type=None, region=None, screen_size=(1920, 1080), **kwargs):
        return REMOTE_SENTINEL
