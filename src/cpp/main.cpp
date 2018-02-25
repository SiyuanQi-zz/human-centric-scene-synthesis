#include "Learner.h"
#include "Sampler.h"
#include "Evaluator.h"
INITIALIZE_EASYLOGGINGPP

void configure_loggers(std::string workspace_path){
    std::string loggerName = LOGGER_NAME;
    //el::Logger* logger = el::Loggers::getLogger(loggerName);

    // Configure logger
    el::Configurations defaultConf;
    defaultConf.setToDefault();
    defaultConf.setGlobally(
            el::ConfigurationType::Filename, workspace_path + "src/logs/" + loggerName + ".log");
    defaultConf.setGlobally(
            el::ConfigurationType::Format, "%datetime %level %msg");
//    defaultConf.setGlobally(
//            el::ConfigurationType::Format, "%level %msg");

//    defaultConf.setGlobally(
//            el::ConfigurationType::ToStandardOutput, "false");
    defaultConf.setGlobally(
            el::ConfigurationType::MaxLogFileSize, "2097152");

//    defaultConf.set(el::Level::Debug,
//                     el::ConfigurationType::Format, "%datetime %level %func %msg");
    defaultConf.set(el::Level::Fatal,
                    el::ConfigurationType::Format, "%datetime %level %func %msg");

    el::Loggers::reconfigureLogger(loggerName, defaultConf);
    CLOG(DEBUG, LOGGER_NAME) << "Logger \"" + loggerName + "\" configured.";
}


int main(int argc, char* argv[]) {
    // Path settings
    std::string suncgRoot = "/home/siyuan/data/SUNCG/";
    std::string workspacePath = "/home/siyuan/projects/release/cvpr2018/";
    std::string metadataPath = workspacePath + "src/metadata/";

    START_EASYLOGGINGPP(argc, argv);
    configure_loggers(workspacePath);
    srand(time(NULL));
    //srand(0);

    clock_t begin = clock();

    // ============================== Learning ==============================
    //FurnitureArranger::Learner learner(suncgRoot, metadataPath, workspacePath);
    //learner.collect_stats();
    //learner.learn_cost_weights();

    //// Write out SUNCG layouts
    //std::string txtOutputFolder = workspacePath + "tmp/suncg/";
    //for (FurnitureArranger::Room room : learner.get_rooms()){
    //    boost::filesystem::create_directories(txtOutputFolder+room.get_caption());
    //    std::string outputFilename = txtOutputFolder + room.get_caption() + "/" + room.get_caption_coarse() + ".txt";
    //    room.write_to_room_arranger_file(outputFilename);
    //}


    //============================== Sample start from SUNCG ==============================
    FurnitureArranger::Learner learner(suncgRoot, metadataPath, workspacePath);
    FurnitureArranger::Sampler Arranger(learner, suncgRoot, metadataPath, workspacePath);
    Arranger.arrange(strtol(argv[1], NULL, 10));


    //============================== Evaluating ==============================
    //FurnitureArranger::Evaluator evaluator(suncgRoot, metadataPath, workspacePath);
    //evaluator.collect_affordance();

    clock_t end = clock();
    std::cout << "Time elapsed: " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    return 0;
}