SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema Medcare
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema Medcare
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `Medcare` DEFAULT CHARACTER SET utf8 ;
USE `Medcare` ;

-- -----------------------------------------------------
-- Table `Medcare`.`Paciente`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Medcare`.`Paciente` (
  `idPaciente` INT NOT NULL AUTO_INCREMENT,
  `nome` VARCHAR(100) NOT NULL,
  `sexo` VARCHAR(1) NOT NULL,
  `data_nascimento` DATE NOT NULL,
  `cpf` VARCHAR(11) NOT NULL,
  `email` VARCHAR(100) NOT NULL,
  `telefone` VARCHAR(12) NOT NULL,
  `cep` VARCHAR(8) NOT NULL,
  `plano` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`idPaciente`),
  UNIQUE INDEX `idPaciente_UNIQUE` (`idPaciente` ASC) VISIBLE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `Medcare`.`Usuario`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Medcare`.`Usuario` (
  `idUsuario` INT NOT NULL AUTO_INCREMENT,
  `nome` VARCHAR(100) NOT NULL,
  `email` VARCHAR(100) NOT NULL,
  `senha` VARCHAR(100) NOT NULL,
  `tipo_usuario` VARCHAR(45) NOT NULL,
  `especialidade` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`idUsuario`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `Medcare`.`Revisao`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Medcare`.`Revisao` (
  `idRevisao` INT NOT NULL AUTO_INCREMENT,
  `comentarios` LONGTEXT NULL,
  `status_revisao` VARCHAR(45) NOT NULL,
  `data_revisao` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`idRevisao`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `Medcare`.`Diagnostico`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Medcare`.`Diagnostico` (
  `idDiagnostico` INT NOT NULL AUTO_INCREMENT,
  `resultado` VARCHAR(100) NOT NULL,
  `probabilidade` FLOAT(0) NOT NULL,
  `data_diagnostico` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`idDiagnostico`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `Medcare`.`Exame`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Medcare`.`Exame` (
  `idExame` INT NOT NULL AUTO_INCREMENT,
  `data_exame` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `imagem_exame` VARCHAR(100) NOT NULL,
  `Revisao_idRevisao` INT NULL,
  `Diagnostico_idDiagnostico` INT NOT NULL,
  `Usuario_idUsuario` INT NOT NULL,
  `Paciente_idPaciente` INT NOT NULL,
  PRIMARY KEY (`idExame`),
  INDEX `fk_Exame_Revisao_idx` (`Revisao_idRevisao` ASC) VISIBLE,
  INDEX `fk_Exame_Diagnostico1_idx` (`Diagnostico_idDiagnostico` ASC) VISIBLE,
  INDEX `fk_Exame_Usuario1_idx` (`Usuario_idUsuario` ASC) VISIBLE,
  INDEX `fk_Exame_Paciente1_idx` (`Paciente_idPaciente` ASC) VISIBLE,
  CONSTRAINT `fk_Exame_Revisao`
    FOREIGN KEY (`Revisao_idRevisao`)
    REFERENCES `Medcare`.`Revisao` (`idRevisao`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_Exame_Diagnostico1`
    FOREIGN KEY (`Diagnostico_idDiagnostico`)
    REFERENCES `Medcare`.`Diagnostico` (`idDiagnostico`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_Exame_Usuario1`
    FOREIGN KEY (`Usuario_idUsuario`)
    REFERENCES `Medcare`.`Usuario` (`idUsuario`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_Exame_Paciente1`
    FOREIGN KEY (`Paciente_idPaciente`)
    REFERENCES `Medcare`.`Paciente` (`idPaciente`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;