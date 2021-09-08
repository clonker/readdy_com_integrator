#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <readdy/common/boundary_condition_operations.h>
#include <readdy/model/actions/UserDefinedAction.h>
#include <readdy/kernel/singlecpu/SCPUKernel.h>

namespace py = pybind11;

class COMIntegrator : public readdy::model::actions::UserDefinedAction {
public:
    explicit COMIntegrator(readdy::scalar timeStep) : UserDefinedAction(timeStep) {}

    void perform() override {
        const auto &context = kernel()->context();
        const auto &pbc = context.periodicBoundaryConditions();

        auto scpuKernel = dynamic_cast<readdy::kernel::scpu::SCPUKernel *>(kernel());
        auto &stateModel = scpuKernel->getSCPUKernelStateModel();
        auto *particleData = stateModel.getParticleData();
        readdy::Vec3 newCOM{0, 0, 0};
        for (auto &entry: *particleData) {
            if (!entry.deactivated) {
                const auto &D = context.particleTypes().diffusionConstantOf(entry.type);
                auto randomDisplacement = readdy::model::rnd::normal3<readdy::scalar>() * sqrt(2. * D * _timeStep);
                auto deterministicDisplacement = entry.force * D * _timeStep / context.kBT();
                entry.pos += randomDisplacement + deterministicDisplacement;
                newCOM += entry.pos;
            }
        }
        newCOM /= static_cast<double>(particleData->size() - particleData->n_deactivated());

        for (auto &entry: *particleData) {
            entry.pos -= newCOM;
            readdy::bcs::fixPosition(entry.pos, context.boxSize(), pbc);
        }
    }
};

PYBIND11_MODULE(bindings, m) {
    py::module_::import("readdy");
    py::class_<COMIntegrator, readdy::model::actions::UserDefinedAction, std::shared_ptr<COMIntegrator>>(m, "COMIntegrator")
            .def(py::init<readdy::scalar>());
}
