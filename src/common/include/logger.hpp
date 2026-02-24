#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <memory>

namespace project_x {

class Logger {
public:
    static std::shared_ptr<spdlog::logger> get() {
        static auto logger = initialize();
        return logger;
    }

private:
    static std::shared_ptr<spdlog::logger> initialize() {
        try {
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_level(spdlog::level::info);
            console_sink->set_pattern("[%H:%M:%S] [%^%l%$] %v");

            auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                "vision_analysis.log", 1024 * 1024 * 10, 3);
            file_sink->set_level(spdlog::level::trace);
            file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%s:%#] %v");

            std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
            auto logger = std::make_shared<spdlog::logger>("vision", sinks.begin(), sinks.end());

            logger->set_level(spdlog::level::trace);
            spdlog::register_logger(logger);

            return logger;
        } catch (const spdlog::spdlog_ex& ex) {
            std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
            return nullptr;
        }
    }
};

#define LOG_TRACE(...)    project_x::Logger::get()->trace(__VA_ARGS__)
#define LOG_DEBUG(...)    project_x::Logger::get()->debug(__VA_ARGS__)
#define LOG_INFO(...)     project_x::Logger::get()->info(__VA_ARGS__)
#define LOG_WARN(...)     project_x::Logger::get()->warn(__VA_ARGS__)
#define LOG_ERROR(...)    project_x::Logger::get()->error(__VA_ARGS__)
#define LOG_CRITICAL(...) project_x::Logger::get()->critical(__VA_ARGS__)

} 

#endif // LOGGER_HPP
