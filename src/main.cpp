#include <tuple>

#include <utility>
#include <string_view>

#include <readdy/readdy.h>
#include <readdy/common/boundary_condition_operations.h>
#include <readdy/model/actions/UserDefinedAction.h>
#include <readdy/kernel/singlecpu/SCPUKernel.h>



template<typename Context, typename Entry, typename scalar>
void normalDiffusion(const Context &context, Entry &entry, scalar dt) {
    const auto &D = context.particleTypes().diffusionConstantOf(entry.type);
    const auto &pbc = context.periodicBoundaryConditions();
    auto randomDisplacement = readdy::model::rnd::normal3<readdy::scalar>() * sqrt(2. * D * dt);
    auto deterministicDisplacement = entry.force * D * dt / context.kBT();
    entry.pos += randomDisplacement + deterministicDisplacement;
}

class COMIntegrator : public readdy::model::actions::UserDefinedAction {
public:
    explicit COMIntegrator(readdy::scalar timeStep) : UserDefinedAction(timeStep) {   }

    template<typename Entry>
    void integrateParticle(Entry &entry, const readdy::model::Context &context) {
        if(constrained(entry.type, begin(constrainedParticleIds), end(constrainedParticleIds))) {
            const auto &pbc = context.periodicBoundaryConditions();
            auto randomDisplacement = readdy::model::rnd::normal3<readdy::scalar>();
            auto deterministicDisplacement = entry.force * _timeStep / context.kBT();
            #pragma unroll
            for(std::uint8_t d = 0; d < 3; ++d) {
                randomDisplacement[d] *= sqrt(2. * anisotropicDiff[d] * _timeStep);
                deterministicDisplacement[d] *= sqrt(anisotropicDiff[d]);
            }
            entry.pos += randomDisplacement + deterministicDisplacement;
            entry.pos[2] = -0.026;
            readdy::bcs::fixPosition(entry.pos, context.boxSize(), pbc);
        } else {
            normalDiffusion(context, entry, _timeStep);
        }
    }

    void perform() override {
        const auto &context = kernel()->context();
        auto scpuKernel = dynamic_cast<readdy::kernel::scpu::SCPUKernel*>(kernel());


	for(auto &entry : *scpuKernel->getSCPUKernelStateModel().getParticleData()) {
		const auto &D = context.particleTypes().diffusionConstantOf(entry.type);
		auto randomDisplacement = readdy::model::rnd::normal3<readdy::scalar>() * sqrt(2. * D * dt);
		auto deterministicDisplacement = entry.force * D * dt / context.kBT();
		entry.pos += randomDisplacement + deterministicDisplacement;
	}
    }

private:
    std::vector<std::string> constrainedParticles;
    std::vector<readdy::ParticleTypeId> constrainedParticleIds;
    AnisotropicDiff anisotropicDiff;
};

PYBIND11_MODULE(diss_bindings, m) {
    py::module_::import("readdy");

    using AniIntegr = AnisotropicIntegrator<readdy::kernel::scpu::SCPUKernel>;
    py::class_<AniIntegr, readdy::model::actions::UserDefinedAction, std::shared_ptr<AniIntegr>> (m, "AnisotropicIntegrator")
        .def(py::init<std::string_view, readdy::scalar, AniIntegr::AnisotropicDiff, std::vector<std::string>>());
}

